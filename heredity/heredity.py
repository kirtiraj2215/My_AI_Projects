import csv
import itertools
import sys

PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {True: 0.65, False: 0.35},
        1: {True: 0.56, False: 0.44},
        0: {True: 0.01, False: 0.99}
    },
    "mutation": 0.01
}

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    probabilities = {
        person: {
            "gene": {2: 0, 1: 0, 0: 0},
            "trait": {True: 0, False: 0}
        }
        for person in people
    }

    names = set(people)
    for have_trait in powerset(names):
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")

def load_data(filename):
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def joint_probability(people, one_gene, two_genes, have_trait):
    probability = 1
    for person in people:
        genes = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )
        has_trait = person in have_trait

        if people[person]["mother"] is None:
            probability *= PROBS["gene"][genes]
        else:
            mother = people[person]["mother"]
            father = people[person]["father"]
            mother_genes = (
                2 if mother in two_genes else
                1 if mother in one_gene else
                0
            )
            father_genes = (
                2 if father in two_genes else
                1 if father in one_gene else
                0
            )
            passing_prob = {
                0: PROBS["mutation"],
                1: 0.5,
                2: 1 - PROBS["mutation"]
            }

            if genes == 2:
                probability *= passing_prob[mother_genes] * passing_prob[father_genes]
            elif genes == 1:
                probability *= (
                    passing_prob[mother_genes] * (1 - passing_prob[father_genes]) +
                    (1 - passing_prob[mother_genes]) * passing_prob[father_genes]
                )
            else:
                probability *= (1 - passing_prob[mother_genes]) * (1 - passing_prob[father_genes])

        probability *= PROBS["trait"][genes][has_trait]

    return probability

def update(probabilities, one_gene, two_genes, have_trait, p):
    for person in probabilities:
        genes = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )
        has_trait = person in have_trait

        probabilities[person]["gene"][genes] += p
        probabilities[person]["trait"][has_trait] += p

def normalize(probabilities):
    for person in probabilities:
        gene_total = sum(probabilities[person]["gene"].values())
        trait_total = sum(probabilities[person]["trait"].values())

        for gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][gene] /= gene_total

        for trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][trait] /= trait_total

if __name__ == "__main__":
    main()
