
from typing import Any, List, Optional
from langchain.agents.agent import AgentExecutor
from langchain.agents import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.memory import ConversationBufferMemory


from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import pandas as pd
import glob
import json
from datetime import datetime


def save_chart(query):
    q_s = ' If any charts or graphs or plots were created save them localy and include the save file names in your response.'
    query += ' . ' + q_s
    return query


def save_uploaded_file():
    df = load_dataframe("data.csv")  # Load the "data.csv" file
    agent = create_pandas_dataframe_agent(ChatOpenAI(model_name='gpt-4o-2024-08-06', temperature=0), df, return_intermediate_steps=True,
                                          save_charts=True, verbose=True)
    return agent, df, ["data.csv"]

def load_dataframe(filename):
    df = pd.read_csv(filename)
    return [df]


def run_query(agent, query_):
    if 'chart' or 'charts' or 'graph' or 'graphs' or 'plot' or 'plt' in query_:
        query_ = save_chart(query_)
    output = agent(query_)
    response, intermediate_steps = output['output'], output['intermediate_steps']
    thought, action, action_input, observation, steps = decode_intermediate_steps(intermediate_steps)
    store_convo(query_, steps, response)
    return response, thought, action, action_input, observation


def decode_intermediate_steps(steps):
    log, thought_, action_, action_input_, observation_ = [], [], [], [], []
    text = ''
    #把thinking process extract出来
    for step in steps:
        thought_.append(':green[{}]'.format(step[0][2].split('Action:')[0]))
        action_.append(':green[Action:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[0]))
        action_input_.append(
            ':green[Action Input:] {}'.format(step[0][2].split('Action:')[1].split('Action Input:')[1]))
        observation_.append(':green[Observation:] {}'.format(step[1]))
        log.append(step[0][2])
        text = step[0][2] + ' Observation: {}'.format(step[1])
    return thought_, action_, action_input_, observation_, text

#chat history
def get_convo():
    convo_file = 'convo_history.json'
    with open(convo_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data, convo_file


def store_convo(query, response_, response):
    data, convo_file = get_convo()
    current_dateTime = datetime.now()
    data['{}'.format(current_dateTime)] = []
    data['{}'.format(current_dateTime)].append({'Question': query, 'Answer': response, 'Steps': response_})

    with open(convo_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
