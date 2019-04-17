from utils import gensim_utils,blacklab, text
import constants
import pdb
from gensim.models import Word2Vec
import argparse

#model.wv.similar_by_word('shake', topn=10, restrict_vocab=100)
#model.predict_output_word(['naked',"nude","undress"],100)

def main(term,distance=5,topn=50):
	model=Word2Vec.load(constants.OUTPUT_FOLDER+'synsets_window_'+str(distance))
	similar_terms=model.wv.most_similar(term,topn=topn)
	
	for i,similar_term in enumerate(similar_terms):
		
		print (str(i)+': '+similar_term[0]+', '+str(similar_term[1]))
	pdb.set_trace()
if __name__ == '__main__':
	


	parser = argparse.ArgumentParser()
	
	parser.add_argument("-d", "--distance", type=int)
	parser.add_argument("-t", "--term", type=str)
	parser.add_argument("-n", "--number", type=int)


	#parser.add_argument('window')
	args = parser.parse_args()

	#parser.add_argument('window')
	args = parser.parse_args()

	if not args.term:
		raise Exception 

	if not args.number:
		args.number = 50

	if not args.distance:
		args.distance = 5
	
	
	main(args.term,args.distance,args.number)

	 
