# Third-party notices

`tests/fixtures/qwen38-template.jinja` reproduces the Qwen3.8-27B chat template
published by the Qwen team / Alibaba Cloud under Apache License 2.0.

Source: https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/blob/main/chat_template.jinja

Upstream model and license: https://huggingface.co/Qwen/Qwen3.8-27B

License text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
The template is used only as a syntax and formatting test fixture. The engine
loads the template belonging to the user's local checkpoint at runtime.

Rust dependencies retain their respective licenses. `Cargo.lock` records the
versions used; no third-party inference engine is embedded.
