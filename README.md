OpenedAI Moderations
--------------------

An OpenAI API compatible moderations server for checking whether text is potentially harmful.

This server is built using [moderation by embeddings](https://huggingface.co/ifmain/moderation_by_embeddings) by [ifmain (Mike Afton)](https://huggingface.co/ifmain) and FastAPI.

This is not affiliated with OpenAI in any way, and no OpenAI API key is required.

Quickstart
----------

Docker (**recommended**):
```shell
docker compose up
```
or:
```shell
apt install git git-lfs
git clone https://huggingface.co/ifmain/moderation_by_embeddings repos/moderation_by_embeddings
pip install -r requirements.txt
python moderations.py --host 127.0.0.1 --port 5002
```

You can use the OpenAI client to interact with the API.
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:5002/v1", api_key='skip')
moderation = client.moderations.create(input="I want to kill them.")
print(moderation.results[0])
```

Links & Documentation
---------------------

- Swagger API docs are available locally via /docs, here: (http://localhost:5002/docs) if you are using the defaults.
- OpenAI Moderations Guide: (https://platform.openai.com/docs/guides/moderation)
- OpenAI Moderations API Reference: (https://platform.openai.com/docs/api-reference/moderations)
- Moderation Model: [moderation by embeddings](https://huggingface.co/ifmain/moderation_by_embeddings) by [ifmain (Mike Afton)](https://huggingface.co/ifmain)
- Embedding model: (https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
