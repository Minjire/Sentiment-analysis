#!/usr/bin/env python
# coding: utf-8

# In[46]:


file_name = "WhatsApp Chat with Dennis Neverest.txt"

messages=[]
count = 0

with open(file_name) as fh:
    for line in fh:
        #remove leading and trailing characters
        line = line.strip()
        #skip blank lines
        if line:
            time, description = line.strip().split('-', 1)
            name, message = description.strip().split(':', 1)
            messages.append({"time":time,"name":name,"message":message.strip()})
            count+=1
        else: continue


print(messages)






