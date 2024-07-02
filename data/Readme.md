## About the Data

To create the Coherent Extractive Summarization Dataset (CESD), we have carefully curated a diverse collection of 1000 instances spanning five distinct categories: (1) News, (2) Debate, (3) TV Show, (4) Meeting, and (5) Dialogue. Within CESD, each category is represented by precisely 200 instances, selected through random sampling from the following datasets:

| Category | Dataset | Link
| ------------- | ------------- | ------------ |
| News | CNN/DM | https://github.com/abisee/cnn-dailymail
| Debate | DebateSum | https://huggingface.co/datasets/Hellisotherpeople/DebateSum
| TV Show | TVR | https://github.com/jayleicn/TVRetrieval/tree/master
| Meeting | MeetingBank | https://meetingbank.github.io/
| Dialogue | DIALOGSum | https://huggingface.co/datasets/knkarthick/dialogsum

You can easily acquire each dataset by accessing the provided link. Once you have successfully downloaded the data, you can proceed to utilize the `src/sampling_data.py` script to perform random sampling on each dataset. 

Upon running this script, you will obtain a set of output files in the form of five .csv files. To make it easy for user, we have included these files in the `/data/sampled_data` folder. Nevertheless, this script can be a valuable pathway for including new dataset for processing (You can simply add new dataset processing function).


### Model Summary Generation using Falcon-40B-Instruct

For the annotation, the objective was two-fold: (1) to create a coherent summary extracting important sentences/phrases from a document that effectively captures the key aspects of the document, and (2) to provide feedback on the steps to go from the model summary to the gold summary. In this process, annotators are provided text document and model summary.

You can use `/data/sampled_data` to acquire text documents across five categories. For generating model summary, we use [Falcon-40B-Instruct](https://huggingface.co/tiiuae/falcon-40b-instruct). We provide a script to generate a model summaries that are used in data annotation process. You can find the script at path `/data/src/generate_model_summary.ipynb`. Please refer to the script for more details.

In this script, just provide the path of `/data/sampled_data` and this script will generate model summaries for you. To ensure the extraction of meaningful sentences for the model summary, and recognizing Falcon's limitations in accurate text generation, we employ the following prompt strategy. Instead of instructing Falcon to generate sentences, we prompt the model to identify and select the most coherent sentences from the document in order to create the summary.

Prompt:
```
You are an extractive summarizer. You are presented with a document. The document is a collection of sentences and each sentence is numbered with sentence ids. Understand the given document and create a meaningful summary by picking sentences from document. Please list the sentence ids as output so that sentences corresponding to the generated ids summarize the document coherently. Learn from the below example:

Document:
1. Olympic gold medallist Jessica Ennis-Hill has confirmed she will return to competition in London this July following her break from athletics to become a mother.
2. Ennis-Hill provided one of London 2012's most captivating storylines by surging to heptathlon gold, and the Sheffield-born star will return to the Olympic Stadium three years on to compete in the Sainsbury's Anniversary Games.
3. The 29-year-old has not competed since the same event in 2013 and gave birth to her son, Reggie, last summer.
4. But her return to action sets up the prospect of a showdown against Katarina Johnson-Thompson, the brilliant young British heptathlete who is the heir to Ennis-Hill's throne.
5. Jessica Ennis-Hill became a national hero when she won heptathlon gold at the London 2012 Olympics.
6. Ennis-Hill has not competed since 2013 as she took time off to become a mother.
7. 'I am really looking forward to it,' said Ennis-Hill, who could compete in the long jump or 110m hurdles.
8. 'My race schedule is starting to starting to take shape and it will be good to compete and get a sense of where I am in my return to competing.
9. 'My main goal this season is to be as competitive as possible with the long-term goal being the Rio Olympics.
10. 'Diamond League meetings always have the best athletes in the world so I\u2019m sure this will be a good test for me; I want to perform well being back on the big stage in London but I will be realistic as 2015 is about the challenge of getting back to competitive shape after having my little boy and ultimately making the necessary progression to be at my best for Rio.'
11. Katarina Johnson-Thompson has emerged as the new rising star of British athletics and Ennis-Hill's heir.
12. Another British hero of London 2012, Mo Farah, will also compete in the Diamond League event, which serves as a warm-up for the World Championships in Beijing in August.
13. Ennis-Hill will take part in the two-day meeting on July 24 and 25, with the Sainsbury's IPC Athletics Grand Prix Final taking place on July 26.
14. Ennis-Hill added: 'The 2012 Olympics were an incredible experience for me and it will be very special to step out on that track again.
15. It will be amazing to compete in front of all our British fans who I am sure will have their own memories of the London Games too.

Summary: <s> [2, 5, 6, 11, 12, 15]

Document: [input text document]

Please Create a concise summary using as few sentences as possible.

Summary: <s>
```

In this context, the input document is provided in sentence format to facilitate the process of coherent sentence selection for the summary using Falcon.
