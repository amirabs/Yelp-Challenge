import json
import matplotlib.pyplot as plt

def main():
    xs = []
    ys = []
    with open('yelp_academic_dataset_business.json') as json_file:
        for line in json_file:
            business = json.loads(line)
            xs.append(business['longitude'])
            ys.append(business['latitude'])
        plt.plot(xs, ys, 'r.', alpha = 0.1, markersize = 0.5)
        plt.axis([-115, -110, 32, 35])
        plt.savefig('out.pdf', format='pdf')

if __name__ == "__main__":
    main()
