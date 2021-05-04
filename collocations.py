import pandas as pd
total_results_freq={}

def collocazioni(post,token,range_w):
	#print(post)
	if token.lower() in post:
		positions=[i for i in range(len(post)) if post[i] == token] #calcola gli indici in cui appare nel messaggio il token
		r_dict={} #inizializza risultati
		for pos in positions:
			for a in range (1,range_w+1):
				if pos+a<len(post):
					try:
						r_dict[post[pos+a]]=r_dict[post[pos+a]]+1
					except:
						r_dict[post[pos+a]]=1
				else:
					pass
				if pos-a>=0:
					try:
						r_dict[post[pos-a]]=r_dict[post[pos-a]]+1
					except:
						r_dict[post[pos-a]]=1
				else:
					pass
		#print(r_dict)
		return r_dict

	else:
		return False

def col_gen(msgs,token,range_w): #input: una colonna di messaggi, il token, e quante parole
	total_results={} #dict con i risultati totali
	total_length=0
	total_posts=0
	msgs=[x for x in msgs if token in x]
	for msg in msgs:
		results=collocazioni(msg,token,range_w)
		if results:#se trovi qualcosa
			total_posts=total_posts+1
			total_length+=len(msg) #aggiungi alla lunghezza totale del sample
			for key,item in results.items(): #updata il dizionario totale con i risultai della funzione
				try:
					total_results[key]=total_results[key]+item
				except:
					total_results[key]=item
	total_results_sort={k: v for k, v in sorted(total_results.items(), key=lambda item: item[1])}
	df=pd.DataFrame.from_dict(total_results,orient="index")
	for	key, item in total_results_sort.items():
		total_results_freq[key]=item/total_length
	df2=pd.DataFrame.from_dict(total_results_freq,orient="index")
	return df, df2

#faccio collocations sul df con le collocazioni
def collocations(words,myseries,length,text_column="processed"): #text column deve essere in token!
	columns=["ITEM"]
	standard_results_row={}
	for x in words:
		columns.append(x)
	for x in columns:
		standard_results_row[x]=0

	#print (standard_results_row)
	results=pd.DataFrame(columns=columns)
	results_rel=results
	for indice_parole, word in enumerate(words):
		print(f"Parola {indice_parole} su {len(words)}")
		mask = myseries[text_column].apply(lambda x: word in x)
		subset=myseries[mask]
		length_subset=len(subset)
		print (f"dataset per {word} is {len(subset)}")
		for index,row in subset.iterrows():
			result_row=standard_results_row
			index_word=row[text_column].index(word)
			length_forward=index_word+length+1
			print(length_forward)
			length_back=index_word-length
			print(length_back)

			if length_back<0:
				length_back=0
			if length_forward+1>len(row[text_column]):
				length_forward=len(row[text_column])
			collocate=row[text_column][length_back:length_forward]
			print (collocate)
			for parola in collocate:
				if parola not in results["ITEM"].values:
					result_row["ITEM"]=parola
					result_row[word]=1
					results=results.append(result_row,ignore_index=True)
					results=results.set_index("ITEM",drop=False)
					print (results)
				else:
					results.at[parola,word]=results.at[parola,word]+1
		results_rel["ITEM"]=results["ITEM"]
		results_rel[word] =results[word].divide(length_subset)
		print(results)
		print(results_rel)

	#results=results.drop(["ITEM"])
	return results,results_rel
