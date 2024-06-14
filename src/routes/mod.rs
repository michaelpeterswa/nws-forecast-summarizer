use axum::extract::{Query, State};
use lazy_static::lazy_static;
use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage},
    Ollama,
};
use prometheus::{opts, register_counter, Counter};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, USER_AGENT};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

use std::sync::Arc;

lazy_static! {
    pub static ref FORECAST_COUNTER: Counter = register_counter!(opts!(
        "forecast_total",
        "times the /api/v1/forecast endpoint was called"
    ))
    .unwrap();
}

#[derive(Clone)]
pub struct ForecastState {
    pub client: reqwest::Client,
    pub ollama_connection: Ollama,
    pub ollama_model: String,
}
// coordinate struct
#[derive(Deserialize)]
struct Coordinates {
    #[serde(rename = "y")]
    latitude: f64,
    #[serde(rename = "x")]
    longitude: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Period {
    pub detailed_forecast: String,
    pub dewpoint: Dewpoint,
    pub end_time: String,
    pub icon: String,
    pub is_daytime: bool,
    pub name: String,
    pub number: i64,
    pub probability_of_precipitation: ProbabilityOfPrecipitation,
    pub relative_humidity: RelativeHumidity,
    pub short_forecast: String,
    pub start_time: String,
    pub temperature: i64,
    pub temperature_trend: Option<String>,
    pub temperature_unit: String,
    pub wind_direction: String,
    pub wind_speed: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Dewpoint {
    pub unit_code: String,
    pub value: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbabilityOfPrecipitation {
    pub unit_code: String,
    pub value: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelativeHumidity {
    pub unit_code: String,
    pub value: i64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimplifiedForecastPeriod {
    pub detailed_forecast: String,
    pub end_time: String,
    pub name: String,
    pub relative_humidity: String,
    pub start_time: String,
    pub temperature: String,
    pub wind_speed: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NShotInOut {
    pub input: String,
    pub output: String,
}

pub async fn root() -> &'static str {
    "nws-forecast-summarizer"
}

pub async fn forecast(
    Query(params): Query<HashMap<String, String>>,
    State(forecast_state): State<Arc<ForecastState>>,
) -> Result<String, &'static str> {
    FORECAST_COUNTER.inc();

    let address_result = match params.get("address") {
        Some(address) => Ok(address.to_owned()),
        None => Err("address parameter is required"),
    };

    let address = match address_result {
        Ok(address) => address,
        Err(e) => return Err(e),
    };

    let coordinates_result = match geocode_address(forecast_state.client.clone(), address).await {
        Ok(coordinates) => Ok(coordinates),
        Err(e) => Err(e.to_string()),
    };

    let coordinates = match coordinates_result {
        Ok(coordinates) => coordinates,
        Err(_) => return Err("error geocoding address"),
    };

    let forecast_url_result =
        match forecast_address_from_coordinates(forecast_state.client.clone(), coordinates).await {
            Ok(forecast_url) => Ok(forecast_url),
            Err(e) => Err(e.to_string()),
        };

    let forecast_url = match forecast_url_result {
        Ok(forecast_url) => forecast_url,
        Err(_) => return Err("error getting forecast URL"),
    };

    let periods_result =
        match get_forecast_periods(forecast_state.client.clone(), forecast_url).await {
            Ok(periods) => Ok(periods),
            Err(e) => Err(e.to_string()),
        };

    let periods = match periods_result {
        Ok(periods) => periods,
        Err(_) => return Err("error getting forecast periods"),
    };

    let mut simplified_forecast_periods: Vec<SimplifiedForecastPeriod> = Vec::new();

    for period in periods {
        simplified_forecast_periods.push(SimplifiedForecastPeriod {
            detailed_forecast: period.detailed_forecast,
            end_time: period.end_time,
            name: period.name,
            relative_humidity: format!("{}%", period.relative_humidity.value),
            start_time: period.start_time,
            temperature: format!("{}{}", period.temperature, period.temperature_unit),
            wind_speed: format!("{} {}", period.wind_speed, period.wind_direction),
        });
    }

    let simplified_forecast_json = serde_json::to_string(&simplified_forecast_periods).unwrap();

    let prompt = "
    You are a tool that can provide concise summaries of weather forecasts.
    Input is a JSON array with one entry per forecast period.
    Output is a JSON object with the key \"summary\" containing the overall forecast in at most four sentences.
    Each entry contains relavant weather information including a detailed text forecast.
    Do not include any information that is not present in the input.
    Do not comment twice on the same weather condition.
    Focus mainly on the daytime periods.
    Avoid editorializing or making assumptions.
    Make the output sound like a human wrote it, with concise but friendly language and complete sentences.
    ";

    let training = vec![NShotInOut {
        input: "[{\"name\": \"Tonight\", \"start_time\": \"2024-06-08T20:00:00-07:00\", \"end_time\": \"2024-06-09T06:00:00-07:00\", \"temperature\": \"54F\", \"detailed_forecast\": \"Mostly cloudy, with a low around 54. East wind around 2 mph.\", \"relative_humidity\": \"80%\", \"wind_speed\": \"2 mph E\"}, {\"name\": \"Sunday\", \"start_time\": \"2024-06-09T06:00:00-07:00\", \"end_time\": \"2024-06-09T18:00:00-07:00\", \"temperature\": \"74F\", \"detailed_forecast\": \"Mostly sunny. High near 74, with temperatures falling to around 72 in the afternoon. Southwest wind 1 to 6 mph.\", \"relative_humidity\": \"79%\", \"wind_speed\": \"1 to 6 mph SW\"}, {\"name\": \"Sunday Night\", \"start_time\": \"2024-06-09T18:00:00-07:00\", \"end_time\": \"2024-06-10T06:00:00-07:00\", \"temperature\": \"51F\", \"detailed_forecast\": \"Mostly cloudy, with a low around 51. West wind 2 to 6 mph.\", \"relative_humidity\": \"85%\", \"wind_speed\": \"2 to 6 mph W\"}, {\"name\": \"Monday\", \"start_time\": \"2024-06-10T06:00:00-07:00\", \"end_time\": \"2024-06-10T18:00:00-07:00\", \"temperature\": \"71F\", \"detailed_forecast\": \"Mostly sunny, with a high near 71. Southwest wind around 3 mph.\", \"relative_humidity\": \"84%\", \"wind_speed\": \"3 mph SW\"}, {\"name\": \"Monday Night\", \"start_time\": \"2024-06-10T18:00:00-07:00\", \"end_time\": \"2024-06-11T06:00:00-07:00\", \"temperature\": \"52F\", \"detailed_forecast\": \"Partly cloudy, with a low around 52. North wind around 3 mph.\", \"relative_humidity\": \"80%\", \"wind_speed\": \"3 mph N\"}, {\"name\": \"Tuesday\", \"start_time\": \"2024-06-11T06:00:00-07:00\", \"end_time\": \"2024-06-11T18:00:00-07:00\", \"temperature\": \"69F\", \"detailed_forecast\": \"Partly sunny, with a high near 69.\", \"relative_humidity\": \"79%\", \"wind_speed\": \"2 to 7 mph SSW\"}, {\"name\": \"Tuesday Night\", \"start_time\": \"2024-06-11T18:00:00-07:00\", \"end_time\": \"2024-06-12T06:00:00-07:00\", \"temperature\": \"50F\", \"detailed_forecast\": \"Mostly cloudy, with a low around 50.\", \"relative_humidity\": \"83%\", \"wind_speed\": \"2 to 7 mph N\"}, {\"name\": \"Wednesday\", \"start_time\": \"2024-06-12T06:00:00-07:00\", \"end_time\": \"2024-06-12T18:00:00-07:00\", \"temperature\": \"67F\", \"detailed_forecast\": \"Mostly sunny, with a high near 67.\", \"relative_humidity\": \"82%\", \"wind_speed\": \"2 to 7 mph NNW\"}, {\"name\": \"Wednesday Night\", \"start_time\": \"2024-06-12T18:00:00-07:00\", \"end_time\": \"2024-06-13T06:00:00-07:00\", \"temperature\": \"47F\", \"detailed_forecast\": \"Mostly clear, with a low around 47.\", \"relative_humidity\": \"86%\", \"wind_speed\": \"1 to 7 mph N\"}, {\"name\": \"Thursday\", \"start_time\": \"2024-06-13T06:00:00-07:00\", \"end_time\": \"2024-06-13T18:00:00-07:00\", \"temperature\": \"69F\", \"detailed_forecast\": \"Mostly sunny, with a high near 69.\", \"relative_humidity\": \"84%\", \"wind_speed\": \"1 to 7 mph N\"}, {\"name\": \"Thursday Night\", \"start_time\": \"2024-06-13T18:00:00-07:00\", \"end_time\": \"2024-06-14T06:00:00-07:00\", \"temperature\": \"49F\", \"detailed_forecast\": \"Partly cloudy, with a low around 49.\", \"relative_humidity\": \"82%\", \"wind_speed\": \"2 to 7 mph NE\"}, {\"name\": \"Friday\", \"start_time\": \"2024-06-14T06:00:00-07:00\", \"end_time\": \"2024-06-14T18:00:00-07:00\", \"temperature\": \"67F\", \"detailed_forecast\": \"A chance of rain after 11am. Partly sunny, with a high near 67.\", \"relative_humidity\": \"81%\", \"wind_speed\": \"2 to 7 mph SW\"}, {\"name\": \"Friday Night\", \"start_time\": \"2024-06-14T18:00:00-07:00\", \"end_time\": \"2024-06-15T06:00:00-07:00\", \"temperature\": \"48F\", \"detailed_forecast\": \"A chance of rain. Mostly cloudy, with a low around 48.\", \"relative_humidity\": \"89%\", \"wind_speed\": \"3 to 7 mph SSW\"}, {\"name\": \"Saturday\", \"start_time\": \"2024-06-15T06:00:00-07:00\", \"end_time\": \"2024-06-15T18:00:00-07:00\", \"temperature\": \"61F\", \"detailed_forecast\": \"A chance of rain. Partly sunny, with a high near 61.\", \"relative_humidity\": \"89%\", \"wind_speed\": \"6 mph SW\"}]".to_string(),
        output: "{\"summary\": \"This week will be mostly sunny and mild, with daytime high temperatures ranging from 61F to 74F. There might be some rain on Friday and Saturday, but it should be light. Humidity will be around 80% to 89%. Winds will be light, mostly from the south and west, up to 7mph.\"}".to_string(),
    }];

    let forecast_system_prompt = ChatMessage::system(prompt.to_string());

    let training_cloned = training.clone();
    let training_user = ChatMessage::user(training_cloned[0].input.to_owned());
    let training_assistant = ChatMessage::assistant(training_cloned[0].output.to_owned());

    let query = ChatMessage::user(simplified_forecast_json);

    let chat = forecast_state
        .ollama_connection
        .send_chat_messages(
            ChatMessageRequest::new(
                forecast_state.ollama_model.clone(),
                vec![
                    forecast_system_prompt,
                    training_user,
                    training_assistant,
                    query,
                ],
            )
            .format(ollama_rs::generation::parameters::FormatType::Json),
        )
        .await
        .unwrap();

    let response = chat.message.unwrap().content;

    Ok(response)
}

async fn geocode_address(
    client: reqwest::Client,
    address: String,
) -> Result<Coordinates, Box<dyn Error>> {
    let census_geocode_url = format!(
        "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress?address={}&benchmark=2020&format=json",
        urlencoding::encode(&address)
    );

    let response_result = client.get(census_geocode_url).send().await;

    let response = match response_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let body_json_result = response.json::<serde_json::Value>().await;

    let body_json = match body_json_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let address_matches_option = body_json["result"]["addressMatches"].as_array();

    let address_matches = match address_matches_option {
        Some(address_matches) => address_matches,
        None => return Err("no address matches found".into()),
    };

    let coordinates_result: Result<Coordinates, serde_json::Error> =
        serde_json::from_str(&address_matches[0]["coordinates"].to_string());

    let coordinates = match coordinates_result {
        Ok(coordinates) => coordinates,
        Err(e) => return Err(e.into()),
    };

    Ok(coordinates)
}

async fn forecast_address_from_coordinates(
    client: reqwest::Client,
    coordinates: Coordinates,
) -> Result<String, Box<dyn Error>> {
    let point_url = format!(
        "https://api.weather.gov/points/{:.5},{:.5}",
        coordinates.latitude, coordinates.longitude
    );

    let mut header_map = HeaderMap::new();
    header_map.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/geojson"),
    );
    header_map.insert(
        USER_AGENT,
        HeaderValue::from_static("nws-forecast-summarizer - michael@michaelpeterswa.com"),
    );

    let point_response_result = client.get(point_url).headers(header_map).send().await;

    let point_response = match point_response_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let point_json_result = point_response.json::<serde_json::Value>().await;

    let point_json = match point_json_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let forecast_url = point_json["properties"]["forecast"].to_string();

    if forecast_url.is_empty() {
        return Err("no forecast URL found".into());
    }

    // unwrap here is safe because the forecast URL is always a string
    let forecast_url_cleaned = forecast_url
        .strip_prefix("\"")
        .unwrap()
        .strip_suffix("\"")
        .unwrap();

    Ok(forecast_url_cleaned.to_string())
}

async fn get_forecast_periods(
    client: reqwest::Client,
    forecast_url: String,
) -> Result<Vec<Period>, Box<dyn Error>> {
    let mut header_map = HeaderMap::new();
    header_map.insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/geojson"),
    );
    header_map.insert(
        USER_AGENT,
        HeaderValue::from_static("nws-forecast-summarizer - michael@michaelpeterswa.com"),
    );

    let forecast_response_result = client.get(forecast_url).headers(header_map).send().await;

    let forecast_response = match forecast_response_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let forecast_json_result = forecast_response.json::<serde_json::Value>().await;

    let forecast_json = match forecast_json_result {
        Ok(body) => body,
        Err(e) => return Err(e.into()),
    };

    let periods_json_string = forecast_json["properties"]["periods"].to_string();
    let periods_json = periods_json_string.as_str();

    if periods_json.is_empty() {
        return Err("no forecast periods found".into());
    }

    let periods_result: Result<Vec<Period>, serde_json::Error> = serde_json::from_str(periods_json);

    let periods = match periods_result {
        Ok(periods) => periods,
        Err(e) => return Err(e.into()),
    };

    Ok(periods)
}
