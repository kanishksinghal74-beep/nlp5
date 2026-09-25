from transformers import pipeline
classifier = pipeline( "sentiment-analysis",  model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",device=-1)

def predict_sentiment(text:str):    
	result = classifier(text)[0]    
	return {
		"label": result["label"],        
		"confidence": float(result["score"])    
    }
