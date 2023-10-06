import xml.etree.ElementTree as ET
import pandas as pd
import re

tree = ET.parse("C:/Users/张铭韬/Desktop/学业/港科大/MSDM5001基础知识/作业/blocklist.xml")
root = tree.getroot()


for child in root:
    print(child.tag, child.attrib)

fatherroots=[]
blockID=[]
id_=[]

for emItem in root.findall('.//{http://www.mozilla.org/2006/addons-blocklist}emItem'):
    attrib = emItem.attrib
    fatherroots.append("emItem")
    blockID.append(attrib["blockID"])
    id_.append(attrib["id"])

for pluginItem in root.findall('.//{http://www.mozilla.org/2006/addons-blocklist}pluginItem'):
    attrib = pluginItem.attrib
    fatherroots.append("pluginItem")
    blockID.append(attrib["blockID"])
    id_.append("")

for gfxBlacklistEntry in root.findall('.//{http://www.mozilla.org/2006/addons-blocklist}gfxBlacklistEntry'):
    attrib = gfxBlacklistEntry.attrib
    fatherroots.append("gfxItem")
    blockID.append(attrib["blockID"])
    id_.append("")


data={"fatherroots":fatherroots,"blockID":blockID,"id":id_}
df=pd.DataFrame(data)
#print(df)

for i in range(df.shape[0]):
  if df.loc[i,"blockID"].startswith("i")!=True or len(df.loc[i,"blockID"])!=4:
    df.drop(i, inplace=True)

df2=df.set_axis(labels=range(df.shape[0]), axis=0)

for i in range(df2.shape[0]):
    print('<%s blockID="%s" id="%s">' %(df2.loc[i,"fatherroots"],df2.loc[i,"blockID"],df2.loc[i,"id"]))



def check(email):
  regex=re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
  if re.fullmatch(regex, email):
    return True
  else:
    return False



df3=pd.DataFrame(data)
for i in range(df3.shape[0]):
  if check(df3.loc[i,"id"])!=True:
    df3.drop(i, inplace=True)

df4=df3.set_axis(labels=range(df3.shape[0]), axis=0)

for i in range(df4.shape[0]):
    print(df4.loc[i,"id"])


