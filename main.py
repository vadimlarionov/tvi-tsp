import math
import random
import matplotlib.pyplot as plt


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def distance_to(self, city):
        x_distance = abs(self.get_x() - city.get_x())
        y_distance = abs(self.get_y() - city.get_y())
        return math.sqrt((x_distance ** 2) + (y_distance ** 2))

    def __repr__(self):
        return str(self.x) + ', ' + str(self.y)


class TourManager(list):
    def load_data(self, filename='data.txt'):
        with open(filename) as f:
            for line in f:
                x, y = map(float, line.split(' '))
                # noinspection PyTypeChecker
                self.append(City(x, y))

    def get_city(self, index):
        return self[index]

    def number_of_cities(self):
        return len(self)


class Tour:
    def __init__(self, tour_manager, tour=None):
        self.tour_manager = tour_manager
        self.tour = []
        self.fitness = 0.0
        self.distance = 0
        if tour is not None:
            self.tour = tour
        else:
            self.tour.extend([None for _ in self.tour_manager])

    def __len__(self):
        return len(self.tour)

    def __getitem__(self, index):
        return self.tour[index]

    def __setitem__(self, key, value):
        self.tour[key] = value

    def __repr__(self):
        gene_string = '|'
        for i in range(0, self.tour_size()):
            gene_string += str(self.get_city(i)) + '|'
        return gene_string

    def generate_individual(self):
        for cityIndex in range(self.tour_manager.number_of_cities()):
            self.set_city(cityIndex, self.tour_manager.get_city(cityIndex))
        random.shuffle(self.tour)

    def get_city(self, tour_position):
        return self.tour[tour_position]

    def set_city(self, tour_position, city):
        self.tour[tour_position] = city
        self.fitness = 0.0
        self.distance = 0

    def get_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.get_distance())
        return self.fitness

    def get_distance(self):
        if self.distance == 0:
            tour_distance = 0
            for cityIndex in range(0, self.tour_size()):
                from_city = self.get_city(cityIndex)
                if cityIndex + 1 < self.tour_size():
                    destination_city = self.get_city(cityIndex + 1)
                else:
                    destination_city = self.get_city(0)
                tour_distance += from_city.distance_to(destination_city)
            self.distance = tour_distance
        return self.distance

    def tour_size(self):
        return len(self.tour)

    def contains_city(self, city):
        return city in self.tour


class Population:
    def __init__(self, tour_manager, population_size, initialise):
        self.tours = []
        for _ in range(population_size):
            self.tours.append(None)

        if initialise:
            for i in range(population_size):
                new_tour = Tour(tour_manager)
                new_tour.generate_individual()
                self.save_tour(i, new_tour)

    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    def save_tour(self, index, tour):
        self.tours[index] = tour

    def get_tour(self, index):
        return self.tours[index]

    def get_fittest(self):
        fittest = self.tours[0]
        for i in range(self.population_size()):
            if fittest.get_fitness() <= self.get_tour(i).get_fitness():
                fittest = self.get_tour(i)
        return fittest

    def population_size(self):
        return len(self.tours)


class GA:
    def __init__(self, tour_manager):
        self.tour_manager = tour_manager
        self.mutationRate = 0.015
        self.tournamentSize = 5
        self.elitism = True

    def evolve_population(self, population):
        new_population = Population(self.tour_manager, population.population_size(), False)
        elitism_offset = 0
        if self.elitism:
            new_population.save_tour(0, population.get_fittest())
            elitism_offset = 1

        for i in range(elitism_offset, new_population.population_size()):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)
            child = self.crossover(parent1, parent2)
            new_population.save_tour(i, child)

        for i in range(elitism_offset, new_population.population_size()):
            self.mutate(new_population.get_tour(i))

        return new_population

    def crossover(self, parent1, parent2):
        child = Tour(self.tour_manager)

        start_pos = int(random.random() * parent1.tour_size())
        end_pos = int(random.random() * parent1.tour_size())

        for i in range(child.tour_size()):
            if start_pos < end_pos and start_pos < i < end_pos:
                child.set_city(i, parent1.get_city(i))
            elif start_pos > end_pos:
                if not start_pos > i > end_pos:
                    child.set_city(i, parent1.get_city(i))

        for i in range(0, parent2.tour_size()):
            if not child.contains_city(parent2.get_city(i)):
                for ii in range(0, child.tour_size()):
                    if child.get_city(ii) is None:
                        child.set_city(ii, parent2.get_city(i))
                        break

        return child

    def mutate(self, tour):
        for tour_pos_1 in range(tour.tour_size()):
            if random.random() < self.mutationRate:
                tour_pos_2 = int(tour.tour_size() * random.random())

                city1 = tour.get_city(tour_pos_1)
                city2 = tour.get_city(tour_pos_2)

                tour.set_city(tour_pos_2, city1)
                tour.set_city(tour_pos_1, city2)

    def tournament_selection(self, pop):
        tournament = Population(self.tour_manager, self.tournamentSize, False)
        for i in range(0, self.tournamentSize):
            random_id = int(random.random() * pop.population_size())
            tournament.save_tour(i, pop.get_tour(random_id))
        fittest = tournament.get_fittest()
        return fittest


def plot(solution):
    x_values = []
    y_values = []
    for point in solution:
        x_values.append(point.get_x())
        y_values.append(point.get_y())
    plt.plot(x_values, y_values, '--ro')
    plt.axis([min(x_values) - 10, max(x_values) + 10, min(y_values) - 10, max(y_values) + 10])
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def __main():
    tour_manager = TourManager()
    tour_manager.load_data()

    # Initialize population
    population = Population(tour_manager, 500, True)
    print('Initial distance: ' + str(population.get_fittest().get_distance()))

    # Evolve population for 100 generations
    ga = GA(tour_manager)
    population = ga.evolve_population(population)
    for _ in range(200):
        population = ga.evolve_population(population)

    print('Finished')
    solution = population.get_fittest()
    print('Solution Distance: ', solution.get_distance())
    plot(solution)


if __name__ == '__main__':
    __main()
