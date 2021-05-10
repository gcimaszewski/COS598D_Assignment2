import torch
import torch.nn as nn
import numpy


class BinOp():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                print(f'need to optimize on {m}')
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets - 2
        self.bin_range = numpy.linspace(start_range,
                                        end_range, end_range-start_range+1)\
                                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []
        self.alpha_matrix = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    print(f'adding model {m}')
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight) #biases are NOT learned
                  #  self.alpha_matrix.append(m.alpha)
        return

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_of_params):
            s = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            # print(f'model here: {self.target_modules[index]} with total {self.num_of_params}')
            # print(f'data for binarizeConvParams {self.target_modules[index].data}')
            if len(s) == 4:
                m = self.target_modules[index].data.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
                # print(f'binarized mean: {m.shape} expand along {s}')
            elif len(s) == 2:
                m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)

            # print(f'parameterizing for dimension {len(s)} with alpha {n}')
            signed_tensor = (self.target_modules[index].data.sign() < 0 )
            self.alpha = m
            self.target_modules[index].data = \
                    self.target_modules[index].data.sign().mul(m.expand(s))

    def restore(self):
        # _copy: copies the elements from src into self tensor and returns self
        for index in range(self.num_of_params):
            # parameter update is done on real-valued weights
            # restore the weights in saved_params back to the layers' data,
            # for optimizer to take step
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            n = weight[0].nelement() # n = c × w × h
            s = weight.size()

            # Now we calculate m, which is the alpha in our notes
            if len(s) == 4:
                m = weight.norm(1, 3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m = weight.norm(1, 1, keepdim=True).div(n).expand(s)

            # Now we calcuate \partial sign(W_i)/ \partial \Tilde{W_i} * alpha. 
            # Note that the following two lines make approximation on \partial sign(W_i)/ \partial \Tilde{W_i}.
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0

            # Now we calculate \partial C/ \partial \Tilde{W_i}
            grad = self.target_modules[index].grad.data

            '''
            Please implement the 2nd term of gradient calculation 
            '''
            # Now we update m as alpha * gradient 
            #m = *** Type your code here *** 
            m = grad.mul(m)
            '''
            End here
            '''

            '''
            Please implement the 1st term of gradient calculation 
            '''
            # Now we calculate m_add, which is defined as sign(W) multiple gradient
            sign_W = weight.sign()
            m_add = grad.multiply(sign_W)
            #m_add = *** Type your code here *** 

            # sum over all the weight entries
            if len(s) == 4:
                m_add = m_add.sum(3, keepdim=True)\
                        .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                m_add = m_add.sum(1, keepdim=True).div(n).expand(s)

            # Now we update m_add as sign(W) * m_add 
            m_add = sign_W.multiply(m_add)
            '''
            End here
            '''
            # Scale the 1st term, and add the 1st and 2nd terms.
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)
