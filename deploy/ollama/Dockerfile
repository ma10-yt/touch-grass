FROM ollama/ollama:0.40.2

COPY deploy/ollama/entrypoint.sh /usr/local/bin/touch-grass-ollama-entrypoint
RUN chmod 0755 /usr/local/bin/touch-grass-ollama-entrypoint

ENV OLLAMA_HOST=0.0.0.0:11434

ENTRYPOINT ["/usr/local/bin/touch-grass-ollama-entrypoint"]
