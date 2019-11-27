
import pandas

OPT_ACT_FILE = 'state_opt_action.csv'

def main():
	cols = ['Health level', 'Number of hearts', 'Wait time', 'Optimal action']
	data = pandas.read_csv(OPT_ACT_FILE, names=cols, dtype=int, header=None)

if __name__ == '__main__':
    main()