from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "I am happy!"

res = classifier(text)

#print(res) # uncomment to see the result in the terminal


# Save the result to a text file
with open('sentiment.txt', 'w') as f:

    f.write("Sentiment Analysis Result:\n")
    f.write("=====================================\n")
    f.write("Text: " + text + "\n")
    f.write("=====================================\n")

    f.write("Results:\n")

    for i in res:
        f.write(f"Label: {i['label']}, \nScore: {i['score']*100} %\n")
