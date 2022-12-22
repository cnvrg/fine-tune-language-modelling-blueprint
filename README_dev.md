You can use this blueprint to fine tune three pre-trained bert/gpt2/distilgpt2 models that can do text generation based on your data.
In order to fine tune these models with your data, you would need to provide a data of stories on different subjects.
For your convenience, you can use one of S3 connector prebuilt datasets, which includes 30k movie stories.
1. Click on `Use Blueprint` button
2. You will be redirected to your blueprint flow page
3. In the flow, after training these models, the model compare library will do models comparision to choose the best model.
4. You can also choose any model to do prediction or use the best model choosen by the system based on the minimum perplexity.
4. Click on the 'Run Flow' button
5. In a few minutes you will fine-tune three popular anguage models to design chatbot and produce a batch prediction result.
6. Go to the 'artifacts' tab in the last batch prediction block, and look for your final result in .csv

Congrats! You have fine-tuned three most popular pre-trained huggingface models that can do stories/articles generation!

[See here how we created this blueprint](https://github.com/cnvrg/fine-tune-language-modelling-blueprint)
