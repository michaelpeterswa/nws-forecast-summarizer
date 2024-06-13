FROM --platform=$BUILDPLATFORM rust:1.78 AS base
RUN cargo install --locked cargo-chef sccache
ENV RUSTC_WRAPPER=sccache SCCACHE_DIR=/sccache

FROM base AS stage-1
WORKDIR /app
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM base as stage-2
WORKDIR /app
COPY --from=stage-1 /app/recipe.json recipe.json
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo chef cook --release --recipe-path recipe.json
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \ 
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=$SCCACHE_DIR,sharing=locked \
    cargo build --release --bin rust-start

# hadolint ignore=DL3006
FROM gcr.io/distroless/cc-debian12 AS stage-3
WORKDIR /app
COPY --from=stage-2 /app/target/release/rust-start /app/rust-start
ENTRYPOINT ["/app/rust-start"]