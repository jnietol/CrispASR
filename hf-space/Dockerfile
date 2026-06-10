FROM ubuntu:22.04
WORKDIR /space

RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get install -y --fix-missing bash curl ffmpeg python3 python3-pip \
      espeak-ng espeak-ng-data libespeak-ng1 && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Pre-built binaries (crispasr + crispasr-lid + shared libs) uploaded to
# the Space repo at bin/. This avoids the 6+ minute C++ compilation that
# exceeds the free-tier Docker build timeout.
COPY bin/crispasr-bin-linux-x64.tar.gz /tmp/
RUN mkdir -p /opt/crispasr/build/bin && \
  tar xzf /tmp/crispasr-bin-linux-x64.tar.gz -C /opt/crispasr/build/bin/ && \
  rm /tmp/crispasr-bin-linux-x64.tar.gz && \
  find /opt/crispasr/build/bin -name 'lib*.so*' -exec cp -a {} /usr/local/lib/ \; && \
  ldconfig

# Fetch bundled sample audio from the GitHub repo (optional).
ARG CRISPASR_REPO=https://github.com/CrispStrobe/CrispASR
RUN mkdir -p /space/samples && \
  curl -sL "${CRISPASR_REPO}/raw/main/samples/jfk.wav" -o /space/samples/jfk.wav 2>/dev/null || true

COPY requirements.txt /space/requirements.txt
RUN pip3 install --no-cache-dir -r /space/requirements.txt

COPY app.py start.sh /space/
RUN chmod +x /space/start.sh && \
  useradd -m -u 1000 app && \
  mkdir -p /models /cache && \
  chown -R app:app /space /opt/crispasr /models /cache

ENV PATH=/opt/crispasr/build/bin:$PATH
ENV CRISPASR_SERVER_URL=http://127.0.0.1:8080
ENV CRISPASR_CACHE_DIR=/cache
ENV CRISPASR_SAMPLES_DIR=/space/samples
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

USER app
EXPOSE 7860
EXPOSE 8080
ENTRYPOINT ["/space/start.sh"]
