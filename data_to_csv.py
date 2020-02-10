filename = input("Filename: ")
with open('data\\'+filename+'.data', 'r') as file:
    with open('data\\csv\\'+filename+'.csv', 'w') as output:

        # population_count = int(file.readline().split()[-1])
        # population_size = int(file.readline().split()[-1])
        # node_cap = int(file.readline().split()[-1])
        # generations = int(file.readline().split()[-1])

        if filename[:8] == 'multiple':
            output.write('node_count, connection_count, accuracy\n')
            i = 0
            for line in file:
                line = line.strip()
                i += 1
                if line[:11] == 'Node Count:':
                    output.write(line.split()[-1] + ',')
                elif line == 'Weighted Connections:':
                    i = 0
                elif line[:13] == 'Best Accuracy':
                    output.write(str(i-1) + ',' + line.split()[-2] + '\n')
        else:
            output.write('node_count, generation, fitness, population\n')
            for line in file:
                line = line.strip()
                if line[:12] == 'Population: ':
                    population = line.split()[1]
                    node_count = line.split()[-1]
                elif line[:13] == 'Generation # ':
                    entries = line.split()
                    output.write(','.join([node_count, entries[2], entries[-1], population]) + '\n')