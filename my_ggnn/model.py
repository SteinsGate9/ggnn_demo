import torch
import torch.nn as nn

class AttrProxy(object):
    """
    Just showing off
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix
    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class Propogator(nn.Module):
    """
    GGNN dense matrix 
    """
    def __init__(self, state_dim, node_num, edge_type_num):
        super(Propogator, self).__init__()

        self.node_num = node_num
        self.edge_type_num = edge_type_num

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Sigmoid()
        )
        self.old_gate = nn.Sequential(
            nn.Linear(state_dim*3, state_dim),
            nn.Tanh()
        )


    def forward(self, state_in, state_out, state_cur, A):
        # [n 2n]
        A_in = A[:, :, :self.node_num*self.edge_type_num]
        # [n 2n]
        A_out = A[:, :, self.node_num*self.edge_type_num:]
        # [n,H]
        a_in = torch.bmm(A_in, state_in)
        # [n,H]
        a_out = torch.bmm(A_in, state_in)
        # [n, 3H]
        a = torch.cat((a_in, a_out, state_cur),2)
        # [n, H]
        r = self.reset_gate(a)
        # [n, H]
        z = self.update_gate(a)
        # [n.3H]
        for_h_hat = torch.cat((a_in, a_out, r*state_cur),2)
        h_hat = self.old_gate(for_h_hat)
        # [n, H]
        output = (1 - z) * state_cur + z * h_hat
        return output




class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()

        self.state_dim = opt.state_dim
        self.annotation_dim = opt.annotation_dim
        self.edge_type_num = opt.edge_type_num
        self.node_num = opt.node_num
        self.n_steps = opt.n_steps

        for i in range(self.edge_type_num):
            # incoming and outgoing edge embedding
            in_fc = nn.Linear(self.state_dim, self.state_dim)
            out_fc = nn.Linear(self.state_dim, self.state_dim)
            self.add_module("in_{}".format(i), in_fc)
            self.add_module("out_{}".format(i), out_fc)

        self.in_fcs = AttrProxy(self, "in_")
        self.out_fcs = AttrProxy(self, "out_")

        # Propogation Model
        self.propogator = Propogator(self.state_dim, self.node_num, self.edge_type_num)

        # Output Model
        # Question here, why there's no sigmoid?
        self.out = nn.Sequential(
            nn.Linear(self.state_dim + self.annotation_dim, self.state_dim),
            nn.Tanh(),
            nn.Linear(self.state_dim, 1)
        )

        self._initialization()


    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


  #  net(init_input, annotation, adj_matrix)
    def forward(self, prop_state, annotation, A):

        for i_step in range(self.n_steps):
            in_states = []
            out_states = []
            for i in range(self.edge_type_num):
                # embedding for different edge_types
                # the reason separating in and out is that
                # to concat them later, basically a trick.
                in_states.append(self.in_fcs[i](prop_state))
                out_states.append(self.out_fcs[i](prop_state))
            in_states = torch.stack(in_states).transpose(0, 1).contiguous()
            in_states = in_states.view(-1, self.node_num*self.edge_type_num, self.state_dim)
            out_states = torch.stack(out_states).transpose(0, 1).contiguous()
            out_states = out_states.view(-1, self.node_num*self.edge_type_num, self.state_dim)
            prop_state = self.propogator(in_states, out_states, prop_state, A)

        join_state = torch.cat((prop_state, annotation), 2)

        output = self.out(join_state)
        output = output.sum(2)
        return output


    def _train(self, epoch, train_dataloader, net, criterion, optimizer, opt):
        net.train()
        for i, (adj_matrix, annotation, target) in enumerate(train_dataloader, 0):
            net.zero_grad()
            padding = torch.zeros(len(annotation), opt.node_num, opt.state_dim - opt.annotation_dim).double()
            init_input = torch.cat((annotation, padding), 2)
            if opt.cuda:
                init_input = init_input.cuda()
                adj_matrix = adj_matrix.cuda()
                annotation = annotation.cuda()
                target = target.cuda()

            pred = net(init_input, annotation, adj_matrix)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            if i % int(len(train_dataloader) / 10 + 1) == 0 and opt.verbal:
                print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(train_dataloader), loss.data[0]))


    def _test(self, test_dataloader, net, criterion, optimizer, opt):
        test_loss = 0
        correct = 0
        net.eval()
        for i, (adj_matrix, annotation, target) in enumerate(test_dataloader, 0):
            padding = torch.zeros(len(annotation), opt.node_num, opt.state_dim - opt.annotation_dim).double()
            init_input = torch.cat((annotation, padding), 2)
            if opt.cuda:
                init_input = init_input.cuda()
                adj_matrix = adj_matrix.cuda()
                annotation = annotation.cuda()
                target = target.cuda()

            output = net(init_input, annotation, adj_matrix)
            test_loss += criterion(output, target).data[0]
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))