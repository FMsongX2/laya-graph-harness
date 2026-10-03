# Upstream components

This repository contains authored integration code. It does not vendor runtime
dependencies, model weights, or paper corpora.

- Laya: https://github.com/NandhaKishorM/laya, pinned source revision
  `fa9a2a7070b1789912a49ae24603bbfb1a78b001`; upstream package declares Apache-2.0.
- Base model: https://huggingface.co/convaiinnovations/laya, tested revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Model access and terms remain upstream.
- Cognee: https://github.com/topoteretes/cognee, tested version 1.6.2.
- Neo4j Community: https://neo4j.com, tested image 5.26.31 with its bundled APOC Core.

Dependencies and upstream assets retain their own licenses. The fetch helper
checks out upstream source with its original license; it does not relicence it.
This notice does not grant a project reuse license. No project license has
been selected yet; upstream licenses remain applicable to their own assets.
