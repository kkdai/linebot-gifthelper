"""
This is a FastAPI app that serves as a webhook for LINE Messenger.
It uses the embedchain library to handle incoming messages and generate appropriate responses.
"""
# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

import os
import sys
from linebot import (
    AsyncLineBotApi, WebhookParser
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient

import aiohttp

from fastapi import Request, FastAPI, HTTPException

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
doc_address = os.getenv('DocAddr', None)
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Document Loader
doc = WebBaseLoader(doc_address)
documents = doc.load()

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Creating embeddings and Vectorization
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Custom Prompts
PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer reply in zh-tw:"""
PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}

# Querying
llm = ChatOpenAI(temperature=0.9, model="gpt-3.5-turbo-0613")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=db.as_retriever(), chain_type_kwargs=chain_type_kwargs)


@app.post("/callback")
async def handle_callback(request: Request):
    """
    Handle the callback from LINE Messenger.

    This function validates the request from LINE Messenger, 
    parses the incoming events and sends the appropriate response.

    Args:
        request (Request): The incoming request from LINE Messenger.

    Returns:
        str: Returns 'OK' after processing the events.
    """
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError as exc:
        raise HTTPException(
            status_code=400, detail="Invalid signature") from exc

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        result = qa({"query": event.message.text})

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result["result"])
        )

    return 'OK'
