INPUT_FILE = 'Data.csv'
import pandas

def read_data():
	data = pandas.read_csv(INPUT_FILE)
	print (data)
	return data

def main():
	data = read_data()
	output = []
	count = 100
	for i in range(count):
		s = None
		a = None
		r = None
		sp = None
		output.append((s,a,r,sp))

if __name__ == '__main__':
	main()