#!/usr/bin/env python
# coding: utf-8

# In[46]:


file_name = "WhatsApp Chat with Dennis Neverest.txt"
dict = {}
count = 0

with open(file_name, encoding="utf8") as fh:
    for line in fh:
        #remove leading and trailing characters
        line = line.strip()
        #skip blank lines
        if line:
            time, description = line.strip().split('-', 1)
            name, message = description.strip().split(':', 1)
            dict[count] = [time,name,message.strip()]
            count+=1
        else: continue


print(dict)


# In[41]:


dict[0][1]


# In[ ]:




