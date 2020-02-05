filename = input("Filename: ")
with open('data\\'+filename+'.data', 'r') as file:
    with open('data\\csv\\'+filename+'.csv', 'w') as output:
        output.write('node_count, generation, fitness, population\n')

        # population_count = int(file.readline().split()[-1])
        # population_size = int(file.readline().split()[-1])
        # node_cap = int(file.readline().split()[-1])
        # generations = int(file.readline().split()[-1])

        for line in file:
            line = line.strip()
            if line[:12] == 'Population: ':
                population = line.split()[1]
                node_count = line.split()[-1]
            elif line[:13] == 'Generation # ':
                entries = line.split()
                output.write(','.join([node_count, entries[2], entries[-1], population]) + '\n')