# Operator must pass a reviewed immutable python:3.12-slim image digest.
ARG PYTHON_IMAGE
FROM ${PYTHON_IMAGE}
USER 10001:10001
WORKDIR /tmp
# No repository, generator, verifier, credentials or held-out data enters image.
ENTRYPOINT []
