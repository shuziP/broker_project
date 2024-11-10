#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 历史数据


# In[2]:


from datetime import datetime
import pandas as pd
import akshare as ak
pd.set_option('display.max_rows', 20)


# In[ ]:





# In[3]:


futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol="c2501", period="1") #period="1"; choice of {"1": "1分钟", "5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟"}
futures_zh_minute_sina_df['symbol'] = "c2501"
futures_zh_minute_sina_df['symbol_name'] = "玉米"
futures_zh_minute_sina_df


# In[4]:


futures_zh_daily_sina_df = ak.futures_zh_daily_sina(symbol="c2501")
# futures_zh_daily_sina_df[-5:]

# df_day= futures_zh_daily_sina_df[-6:-1]
df_day = futures_zh_daily_sina_df[-5:]
df_day


# In[5]:


df_day_string = ""

# 遍历DataFrame的行
for _, row in df_day[['date', 'close']].iterrows():
    # 将行的值转换为字符串并拼接起来
    # 这里我们假设你想要一个制表符分隔的格式，但你可以根据需要更改
    line = "价格：\t".join(map(str, row.values)) + "\n"
    # 将这一行添加到结果字符串中
    df_day_string += line

# 去除末尾可能存在的多余换行符（如果最后一行也添加了换行符的话）
if df_day_string.endswith("\n"):
    df_day_string = df_day_string[:-1]

# 打印字符串以验证
print(df_day_string)




# In[6]:


# https://so.eastmoney.com/news/s
stock_news_em_df = ak.stock_news_em(symbol="玉米")
stock_news_em_df


# In[ ]:





# In[7]:


# 获取当前系统时间
now = datetime.now()
 
# 将时间转换为字符串格式，例如 '%Y-%m-%d %H:%M:%S'
time_str = now.strftime('%Y-%m-%d %H:%M:%S')
today_str = time_str[0:10] # 获取系统
star_1 = ' 09:30:00' 
end_2 = ' 10:00:00'

start_time = today_str + star_1
end_time = today_str + end_2
print(start_time)
print(end_time)


# In[8]:


# 假设futures_zh_minute_sina_df是已经加载到内存中的DataFrame
# 并且datetime字段是字符串格式的日期时间

# 将datetime列转换为pandas的datetime类型
futures_zh_minute_sina_df['datetime'] = pd.to_datetime(futures_zh_minute_sina_df['datetime'])

# 定义时间范围
# start_time = '2024-11-08 09:30:00'
# end_time = '2024-11-08 10:00:00'

# 将字符串转换为datetime类型
start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)

# 筛选数据
filtered_df = futures_zh_minute_sina_df[(futures_zh_minute_sina_df['datetime'] >= start_time) & 
                                      (futures_zh_minute_sina_df['datetime'] <= end_time)]

filtered_df['datetime'] = filtered_df['datetime'].dt.strftime('%H:%M')
# 打印筛选后的数据
filtered_df[['datetime','close']]


# In[9]:


# 假设filtered_df是你的DataFrame
# 初始化一个空字符串来存储结果
result_string = ""

# 遍历DataFrame的行
for _, row in filtered_df[['datetime', 'close']].iterrows():
    # 将行的值转换为字符串并拼接起来
    # 这里我们假设你想要一个制表符分隔的格式，但你可以根据需要更改
    line = "价格：\t".join(map(str, row.values)) + "\n"
    # 将这一行添加到结果字符串中
    result_string += line

# 去除末尾可能存在的多余换行符（如果最后一行也添加了换行符的话）
if result_string.endswith("\n"):
    result_string = result_string[:-1]

# 打印字符串以验证
print(result_string)

# 现在你可以将这个字符串传入大模型进行对话了


# In[10]:


new_df = stock_news_em_df[['发布时间', '新闻标题', '新闻内容' ]].head(7)
new_df


# In[11]:


# 假设filtered_df是你的DataFrame
# 初始化一个空字符串来存储结果
news_string = ""

# 遍历DataFrame的行
for _, row in new_df.iterrows():
    # 将行的值转换为字符串并拼接起来
    # 这里我们假设你想要一个制表符分隔的格式，但你可以根据需要更改
    line = "\t".join(map(str, row.values)) + "\n"
    print(line)
    # 将这一行添加到结果字符串中
    news_string += line

# 去除末尾可能存在的多余换行符（如果最后一行也添加了换行符的话）
if news_string.endswith("\n"):
    news_string = news_string[:-1]

# 打印字符串以验证
print(news_string)

# 现在你可以将这个字符串传入大模型进行对话了


# In[12]:


news_string


# In[13]:


import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

file_path = r'D:\project\zp_llm_key.txt'    #apikey存放地址


try:  
    with open(file_path, 'r', encoding='utf-8') as file:  
        openai_api_key = file.read() 
except FileNotFoundError:  
    print(f"文件 {file_path} 未找到，请检查路径是否正确。")

def call_glm(model="glm-4-flash", temperature=0.85, openai_api_key=openai_api_key, user_input="你好",system_input = '假如你是一个资深期货股票交易员请阅读今日开盘30分钟的历史数据和最近新闻数据，分析当前市场趋势并提供交易信号，输出今日是否应该开仓的判断，并详细说明你的原因：'):
    if not openai_api_key:
        print("API key is missing.")
        return "API key is missing."

    if not user_input.strip():
        print("User input cannot be empty.")
        return "User input cannot be empty."

    print(f"正在调用模型: {model}，温度设定: {temperature}")

    try:
        llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            openai_api_key=openai_api_key,
            openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
        )

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    system_input
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )

        messages = [
            ("system", system_input),
            ("human", user_input),
        ]

        # 调用模型
        ai_msg = llm.invoke(messages)
        print("模型返回结果:", ai_msg)

        return ai_msg.content

    except Exception as e:
        # 捕获所有异常并返回错误信息
        error_message = f"模型调用出错: {str(e)}"
        print(error_message)
        return error_message

# if __name__ == '__main__':
#     call_glm(user_input= "你能介绍一下你自己吗？")


# In[14]:


txt_input =news_string
new_summarize = call_glm(user_input= txt_input,system_input ='假如你是一个资深的财经资讯分析师，请你阅读下面的玉米相关的财经新闻，总结和玉米期货相关的部分，剔除股票相关的新闻，形成总结')
print(new_summarize)


# In[15]:


txt_input = '近7日玉米期货开盘日维度价格数据：' + df_day_string+ '\n今日玉米期货开盘三十分钟价格数据：' + result_string+ '\n近期玉米期货新闻：'+new_summarize



# In[16]:


txt_input


# In[17]:


llm_ouptput = call_glm(user_input= txt_input,system_input ='假如你是一个资深的日内短线期货交易员，请你请阅读今日开盘30分钟的历史数据和最近新闻数据，设置一个今日的开仓买入价格和平仓价格以及止损价格', temperature=0.1)
print(llm_ouptput)


import json

def load_config(filename):
    """从指定的 JSON 文件加载配置"""
    try:
        with open(filename, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print("错误：找不到配置文件。")
        return {}
    except json.JSONDecodeError:
        print("错误：无法解析 JSON 格式的配置文件。")
        return {}


from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

import time
import platform

def qq_sendmail(mail_title = 'Python自动发送的邮件',mail_content = "您好，这是使用python登录QQ邮箱发送邮件的测试——zep"):
    # 加载配置
    config = load_config('D:\project\email_config.json')

    if 'host_server' in config and 'sender_qq' in config and 'pwd' in config:
        host_server = config['host_server']
        sender_qq = config['sender_qq']
        pwd = config['pwd']
        receiver = config['receiver']
    else:
        # 如果没有找到这些配置项，可以设置默认值
        print("email_config缺失")

    now = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    host_server = 'smtp.qq.com'  #qq邮箱smtp服务器
    sender_qq = sender_qq #发件人邮箱
    pwd = pwd
    receiver = [receiver ]#收件人邮箱
    mail_title = mail_title #邮件标题
    mail_content = mail_content #邮件正文内容

    # mail_content_time = mail_content +"\n" +now+'\n\n来自'+platform_str+"_"+platform_node_str
    mail_content_time = mail_content +"\n" +now+'\n'
    # 初始化一个邮件主体
    msg = MIMEMultipart()
    msg["Subject"] = Header(mail_title,'utf-8')
    msg["From"] = sender_qq
    # msg["To"] = Header("测试邮箱",'utf-8')
    msg['To'] = ";".join(receiver)
    # 邮件正文内容
    msg.attach(MIMEText(mail_content_time,'plain','utf-8'))



    smtp = SMTP_SSL(host_server) # ssl登录

    # login(user,password):
    # user:登录邮箱的用户名。
    # password：登录邮箱的密码，像笔者用的是网易邮箱，网易邮箱一般是网页版，需要用到客户端密码，需要在网页版的网易邮箱中设置授权码，该授权码即为客户端密码。
    smtp.login(sender_qq,pwd)

    # sendmail(from_addr,to_addrs,msg,...):
    # from_addr:邮件发送者地址
    # to_addrs:邮件接收者地址。字符串列表['接收地址1','接收地址2','接收地址3',...]或'接收地址'
    # msg：发送消息：邮件内容。一般是msg.as_string():as_string()是将msg(MIMEText对象或者MIMEMultipart对象)变为str。
    smtp.sendmail(sender_qq,receiver,msg.as_string())

    # quit():用于结束SMTP会话。
    smtp.quit()

    return 0

qq_sendmail(mail_title = 'Python自动发送的期货设置',mail_content =llm_ouptput)