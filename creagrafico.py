import networkx as nx


def crea_grafico(df):
	colonne= df.columns
	print(colonne)
	#colonne.remove('ITEM').remove('ITEM')
	graph=nx.Graph()
	print(df.index)
	graph.add_nodes_from(list(df.index))
	graph.add_nodes_from(list(colonne))
	for index, row in df.iterrows():
		for colonna in colonne:
			#if type(row[colonna]) == float:
			if row[colonna] > 0:
				graph.add_edge(index, colonna, weight=row[colonna])

	return graph


if __name__=="__main__": #se usato standalone, fai questo (inutile, ma solo per darti idea di cosa fare)
	nx.write_gexf(crea_grafico(df), "NOMEDELGRAFICO.gexf")
