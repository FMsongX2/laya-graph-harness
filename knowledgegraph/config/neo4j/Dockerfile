FROM neo4j@sha256:d9cfe82983d27f5a75b3aaae8f316d04f9a698a3b7f6103a508f7caf8362f255
USER root
RUN mkdir -p /plugins && cp /var/lib/neo4j/labs/apoc-5.26.31-core.jar /plugins/apoc.jar && chown -R neo4j:neo4j /plugins
ENV NEO4J_server_memory_heap_initial__size=256m \
    NEO4J_server_memory_heap_max__size=512m \
    NEO4J_server_memory_pagecache_size=256m \
    NEO4J_dbms_security_procedures_allowlist=apoc.* \
    NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    NEO4J_dbms_usage__report_enabled=false
