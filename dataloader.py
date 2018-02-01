#!/usr/bin/env python3

def load_data(filename):
    data = ""
    with open(filename, 'r') as f:
        data = f.read()
    return data

def main():
    data = load_data("data/input.txt")
    print(data)

if __name__ == "__main__":
    main()

