import time


class Random(object):
    def __init__(self, seed=0):
        self.seed = seed
        self.start_no = 0xabfeeed

    def generate_seed(self):
        g = time.time()
        return int((g - int(g))*10000000000)
    
    def random(self):
        if self.seed == 0:
            self.seed = self.generate_seed()
        random_number = self.seed & 0xfffffffffff
        random_number ^= ( random_number >> 25 )
        random_number ^= ( random_number << 13 )
        random_number *= 0x2545 * 0.00000000001
        return random_number - int(random_number)
        
    def randint(self, start, end):
        if self.seed == 0:
            self.seed = self.generate_seed()
        random_number = self.seed & 0xfffffffffff
        random_number ^= ( random_number >> 25 )
        random_number ^= ( random_number << 13 )
        random_number *= self.generate_seed() * 0.00000000001
        random_number = random_number - int(random_number)
        random_number =  int(random_number*100000)
        random_number = (random_number % (end-start)) + start
        return random_number