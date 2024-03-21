OpenedAI Moderations
--------------------

An OpenAI API compatible moderations server for checking whether text is potentially harmful.

This is a low effort implementation using embeddings for moderation scores that mostly works. It depends on an existing OpenAI Embeddings API compatible server, such as huggingface's [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference). Included is a docker-compose file for running both Moderations and Embeddings together.

To run:
Set the OPENAI_API_BASE environment variable to point to the moderations server.

Or like so: 
```python
import openai

client = openai.Client(base_url="http://127.0.0.1:5002/v1")
response = client.moderations.create(input=["Some good words", "Some bad words"])
for r in response.results:
    if r.flagged:
        print(r)
```

Some adjustment of the `scale_factor` and `flag_threshold` maybe required for your application, edit them in `moderations.env`.

API Documentation
-----------------

- Local API docs are available via <API>/docs, here: (http://localhost:5002/docs) if you are using the defaults.
- OpenAI Moderations Guide: https://platform.openai.com/docs/guides/moderation
- OpenAI Moderations API Reference: https://platform.openai.com/docs/api-reference/moderations

Bugs
----

- It's a simple implementation, so it doesn't have the full power of a trained model
- The Docker included text-embeddings-inference server uses the wrong openai api base url, see: https://github.com/huggingface/text-embeddings-inference/issues/213, this is worked around in moderations.env