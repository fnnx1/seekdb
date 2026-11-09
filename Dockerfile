# MiniVectorDB evaluation image.
# Build from the submitted source; no prebuilt binary is relied on.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ make ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Makefile ./
COPY src ./src
RUN make

RUN mkdir -p /data
VOLUME ["/data"]

# The evaluation driver connects via stdin/stdout and uses -i (no -t/-d).
ENTRYPOINT ["/app/minivectordb"]
