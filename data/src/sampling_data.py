import json
import pandas as pd
import glob
from sklearn.utils import shuffle
import os
from os.path import join

from datasets import load_dataset
from transformers import AutoTokenizer

model = "tiiuae/falcon-40b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
prompt= "You are an extractive summarizer. You are presented with a document. The document is a collection of sentences and each sentence is numbered which is called sentence ids. Understand the given document and create a meaningful summary by picking the sentence from document. Please list the sentence ids hence the generated paragraph (by corresponding sentence) summarize the document coherently. Learn from the below example:\r\n\r\nDocument:\r\n1. Olympic gold medallist Jessica Ennis-Hill has confirmed she will return to competition in London this July following her break from athletics to become a mother.\r\n2. Ennis-Hill provided one of London 2012's most captivating storylines by surging to heptathlon gold, and the Sheffield-born star will return to the Olympic Stadium three years on to compete in the Sainsbury's Anniversary Games.\r\n3. The 29-year-old has not competed since the same event in 2013 and gave birth to her son, Reggie, last summer.\r\n4. But her return to action sets up the prospect of a showdown against Katarina Johnson-Thompson, the brilliant young British heptathlete who is the heir to Ennis-Hill's throne.\r\n5. Jessica Ennis-Hill became a national hero when she won heptathlon gold at the London 2012 Olympics .\r\n6. Ennis-Hill has not competed since 2013 as she took time off to become a mother .\r\n7. 'I am really looking forward to it,' said Ennis-Hill, who could compete in the long jump or 110m hurdles.\r\n8. 'My race schedule is starting to starting to take shape and it will be good to compete and get a sense of where I am in my return to competing.\r\n9. 'My main goal this season is to be as competitive as possible with the long-term goal being the Rio Olympics.\r\n10. 'Diamond League meetings always have the best athletes in the world so I\u2019m sure this will be a good test for me; I want to perform well being back on the big stage in London but I will be realistic as 2015 is about the challenge of getting back to competitive shape after having my little boy and ultimately making the necessary progression to be at my best for Rio.'\r\n11. Katarina Johnson-Thompson has emerged as the new rising star of British athletics and Ennis-Hill's heir .\r\n12. Another British hero of London 2012, Mo Farah, will also compete in the Diamond League event, which serves as a warm-up for the World Championships in Beijing in August.\r\n13. Ennis-Hill will take part in the two-day meeting on July 24 and 25, with the Sainsbury's IPC Athletics Grand Prix Final taking place on July 26.\r\n14. Ennis-Hill added: 'The 2012 Olympics were an incredible experience for me and it will be very special to step out on that track again.\r\n15. It will be amazing to compete in front of all our British fans who I am sure will have their own memories of the London Games too.\r\n\r\nSummary: <s> [1, 2, 4, 7, 10, 11, 13, 14, 15]\r\n\r\nDocument:\r\n"

#path where you want to save data
save_data="/Users/mihirp/Documents/my_project/datasets/processed_data/full_data"

if not os.path.exists(save_data):
    os.makedirs(save_data)

def combine_all():

    file_lists= glob.glob(join(save_data, "*.csv"))

    final_data=[]
    for file in file_lists:
        filename= file.split('/')[-1][:-4]
        print(filename)

        if filename=='cnndm':
            type_name= 'News Article'
        elif filename=='tvqa':
            type_name= 'TV Show'
        elif filename=='dialogsum':
            type_name= 'Dialogue'
        elif filename=='debatesum':
            type_name= 'Debate'
        elif filename=='meetingbank':
            type_name= 'Meeting'

        data_df= pd.read_csv(file)

        for index, row in data_df.iterrows():
            new_data= {'type': type_name, 'source_text': row.input} 
            final_data.append(new_data)

    final_data_df= pd.DataFrame(final_data)

    return final_data_df

def process_meeting(samples):

    #path to data files

    with open("/Users/mihirp/Documents/my_project/datasets/meeting/MeetingBank/Metadata/MeetingBank.json", 'r') as f:
        data= json.load(f)
    
    final_data=[]
    for key, value in data.items():
        for k, v in value['itemInfo'].items():
            ip_text=str()
            for transcript in v['transcripts']:
                ip_text += 'Speaker ' + str(transcript['speaker']) + ": " + transcript['text'] + '\n'

            ip = prompt + ip_text + '\nSummary: <s> '
            tokenized_text= tokenizer(ip)

            if len(tokenized_text['input_ids']) < 2048:
                final_data.append(ip_text)   
    
    data_df= pd.DataFrame(final_data, columns=['input'])
    final_data_df= data_df.sample(n=samples)

    return final_data_df

def process_debate(samples):

    #path to data files
    data_df= pd.read_csv('/Users/mihirp/Documents/my_project/datasets/debate/DebateSumV3.csv', low_memory=False)
    data_df= shuffle(data_df)

    final_data=[]
    for index, row in data_df.iterrows():
        ip = prompt + row['Full-Document'] + '\nSummary: <s> '
        tokenized_text= tokenizer(ip)

        if len(tokenized_text['input_ids']) < 2048:
            final_data.append(row['Full-Document'])
        
        if len(final_data)==samples:
            print(index)
            break

    final_data_df= pd.DataFrame(final_data, columns=['input'])

    return final_data_df

def process_dialog(samples):

    #path to data files
    data_df= pd.read_csv("/Users/mihirp/Documents/my_project/datasets/dialogue/dialogsum.csv")
    data_df= shuffle(data_df)
    data_df= data_df.drop(columns=['summary', 'topic', 'id'])

    final_data=[]
    for index, row in data_df.iterrows():
        ip = prompt + row.dialogue + '\nSummary: <s> '
        tokenized_text= tokenizer(ip)

        if len(tokenized_text['input_ids']) < 2048:
            final_data.append(row.dialogue)
        
        if len(final_data)==samples:
            print(index)
            break

    final_data_df= pd.DataFrame(final_data, columns=['input'])

    return final_data_df

def process_tvqa(samples):

    #path to data files
    data_df= pd.read_json('/Users/mihirp/Documents/my_project/datasets/video_transcripts/tvqa_preprocessed_subtitles.jsonl', lines=True)
    data_df= shuffle(data_df)
    
    final_data=[]
    for index, row in data_df.iterrows():
        ip_text=str()
        for element in row['sub']:
            ip_text+=element['text']+' '
        
        ip = prompt + ip_text + '\nSummary: <s> '
        tokenized_text= tokenizer(ip)

        if len(tokenized_text['input_ids']) < 2048:
            final_data.append(ip_text)
        
        if len(final_data)==samples:
            print(index)
            break
    
    final_data_df= pd.DataFrame(final_data, columns=['input'])

    return final_data_df

def process_cnndm(samples):

    #path to data files
    data_df= pd.read_csv('/Users/mihirp/Documents/my_project/datasets/news/cnndm/test.csv')
    data_df= shuffle(data_df)

    data_df = data_df.drop(columns=['id', 'highlights'])

    final_data=[]

    for index, row in data_df.iterrows():
        ip = prompt + row.article + '\nSummary: <s> '
        tokenized_text= tokenizer(ip)

        if len(tokenized_text['input_ids']) < 2048:
            final_data.append(row.article)
        
        if len(final_data)==samples:
            print(index)
            break

    final_data_df= pd.DataFrame(final_data, columns=['input'])

    return final_data_df

def main():
    result_df_cnndm= process_cnndm(samples=200)
    result_df_tvqa= process_tvqa(samples=200)
    result_df_dialog= process_dialog(samples=200)
    result_df_debate= process_debate(samples=200)
    result_df_meeting= process_meeting(samples=200)

    result_df_cnndm.to_csv(join(save_data, "cnndm.csv"), index=False)
    result_df_tvqa.to_csv(join(save_data, "tvqa.csv"), index=False)
    result_df_dialog.to_csv(join(save_data, "dialogsum.csv"), index=False)
    result_df_debate.to_csv(join(save_data, "debatesum.csv"), index=False)
    result_df_meeting.to_csv(join(save_data, "meetingbank.csv"), index=False)

    result_df= combine_all()

    result_df.to_csv(join(save_data, 'combined_all.csv'), index=False)

if __name__=="__main__":
    main()