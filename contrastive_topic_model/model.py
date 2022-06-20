import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

class ContBertTopicExtractorAE(nn.Module):
    def __init__(self, N_topic, N_word, bert, bert_dim=384, n_hidden=128, hidden_sizes=(100, 100), dropout=0.2):
        super(ContBertTopicExtractorAE, self).__init__()
        self.dim = bert_dim
        self.n_hidden = n_hidden
        self.N_topic = N_topic
        self.N_word = N_word
        self.encoder = AutoModel.from_pretrained(bert)
        self.dim = self.encoder.embeddings.word_embeddings.embedding_dim
        self.fc = nn.Linear(self.dim, self.N_topic)
        
        # Decoder part
        self.adapt_bert = nn.Linear(self.dim, hidden_sizes[0])
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout)
        self.drop_theta = nn.Dropout(p=dropout)
        
        self.decode_fc = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
        
        self.f_mu = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_mu_batchnorm = nn.BatchNorm1d(N_topic, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_sigma_batchnorm = nn.BatchNorm1d(N_topic, affine=False)
        
        self.beta = torch.Tensor(self.N_topic, self.N_word)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(self.N_word, affine=False)
        
        # Variable for KL divergence
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * N_topic)
        #if torch.cuda.is_available():
        #    self.prior_mean = self.prior_mean.cuda()
        self.prior_mean = nn.Parameter(self.prior_mean)
        
        topic_prior_variance = 1. - (1. / self.N_topic)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * N_topic)
        #if torch.cuda.is_available():
        #    self.prior_variance = self.prior_variance.cuda()
        self.prior_variance = nn.Parameter(self.prior_variance)
                
        
    def decode(self, embedding):
        x = self.adapt_bert(embedding)
        x = self.activation(x)
        x = self.decode_fc(x)
        x = self.dropout(x)

        topic_logit = self.f_mu_batchnorm(self.f_mu(x))
        latent_topic = F.softmax(topic_logit, dim=1)
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
        return word_dist, topic_logit
        
        
    def forward(self, input_ids, attention_mask, return_topic=False):
        if return_topic:
            output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
            embedding = output['pooler_output']
            logit = self.fc(embedding)
            latent_topic = F.softmax(logit, dim=1)
            return latent_topic, F.normalize(embedding, dim=1)
        else:
            with torch.no_grad():
                output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
                embedding = output['pooler_output']
            x = self.adapt_bert(embedding)
            x = self.activation(x)
            x = self.decode_fc(x)
            x = self.dropout(x)
        
            topic_logit = self.f_mu_batchnorm(self.f_mu(x))
            latent_topic = F.softmax(topic_logit, dim=1)
            word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
            return word_dist, topic_logit


class ContBertTopicExtractorv2(nn.Module):
    def __init__(self, N_topic, N_word, bert, bert_dim=384, n_hidden=128, hidden_sizes=(100, 100), dropout=0.2):
        super(ContBertTopicExtractorv2, self).__init__()
        self.dim = bert_dim
        self.n_hidden = n_hidden
        self.N_topic = N_topic
        self.N_word = N_word
        self.encoder = AutoModel.from_pretrained(bert)
        self.fc = nn.Linear(self.dim, self.N_topic)
        
        # Decoder part
        self.adapt_bert = nn.Linear(self.dim, hidden_sizes[0])
        self.input_layer = nn.Linear(hidden_sizes[0]+N_word, hidden_sizes[0])
        
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout)
        self.drop_theta = nn.Dropout(p=dropout)
        
        self.decode_fc = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
        
        self.f_mu = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_mu_batchnorm = nn.BatchNorm1d(N_topic, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_sigma_batchnorm = nn.BatchNorm1d(N_topic, affine=False)
        
        self.beta = torch.Tensor(self.N_topic, self.N_word)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(self.N_word, affine=False)
        
        # Variable for KL divergence
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * N_topic)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        self.prior_mean = nn.Parameter(self.prior_mean)
        
        topic_prior_variance = 1. - (1. / self.N_topic)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * N_topic)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        self.prior_variance = nn.Parameter(self.prior_variance)
        
        
    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
        
    def decode(self, bow, embedding):
        x = self.adapt_bert(embedding)
        x = torch.cat((bow, x), 1)
        x = self.input_layer(x)        
        x = self.activation(x)
        x = self.decode_fc(x)
        x = self.dropout(x)

        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        topic_logit = self.reparameterize(mu, log_sigma)
        latent_topic = F.softmax(topic_logit, dim=1)
        latent_topic = self.drop_theta(latent_topic)
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
        return word_dist, topic_logit, mu, log_sigma
        
        
    def forward(self, input_ids, attention_mask, return_topic=False):
        if return_topic:
            output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
            embedding = output['pooler_output']
            logit = self.fc(embedding)
            latent_topic = F.softmax(logit, dim=1)
            return latent_topic, F.normalize(embedding, dim=1)
        else:
            with torch.no_grad():
                output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
                embedding = output['pooler_output']
            x = self.adapt_bert(embedding)
            x = self.activation(x)
            x = self.decode_fc(x)
            x = self.dropout(x)
        
            mu = self.f_mu_batchnorm(self.f_mu(x))
            log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
            topic_logit = self.reparameterize(mu, log_sigma)
            latent_topic = F.softmax(topic_logit, dim=1)
            latent_topic = self.drop_theta(latent_topic)
            word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
            return word_dist, topic_logit, mu, log_sigma





class ContBertTopicExtractor(nn.Module):
    def __init__(self, N_topic, N_word, bert, bert_dim=384, n_hidden=128, hidden_sizes=(100, 100), dropout=0.2):
        super(ContBertTopicExtractor, self).__init__()
        self.dim = bert_dim
        self.n_hidden = n_hidden
        self.N_topic = N_topic
        self.N_word = N_word
        self.encoder = AutoModel.from_pretrained(bert)
        self.dim = self.encoder.embeddings.word_embeddings.embedding_dim
        self.fc = nn.Linear(self.dim, self.N_topic)
        
        # Decoder part
        self.adapt_bert = nn.Linear(self.dim, hidden_sizes[0])
        self.activation = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout)
        self.drop_theta = nn.Dropout(p=dropout)
        
        self.decode_fc = nn.Sequential(OrderedDict([
            ('l_{}'.format(i), nn.Sequential(nn.Linear(h_in, h_out), self.activation))
            for i, (h_in, h_out) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:]))]))
        
        self.f_mu = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_mu_batchnorm = nn.BatchNorm1d(N_topic, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], N_topic)
        self.f_sigma_batchnorm = nn.BatchNorm1d(N_topic, affine=False)
        
        self.beta = torch.Tensor(self.N_topic, self.N_word)
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)
        self.beta_batchnorm = nn.BatchNorm1d(self.N_word, affine=False)
        
        # Variable for KL divergence
        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor(
            [topic_prior_mean] * N_topic)
        #if torch.cuda.is_available():
        #    self.prior_mean = self.prior_mean.cuda()
        self.prior_mean = nn.Parameter(self.prior_mean)
        
        topic_prior_variance = 1. - (1. / self.N_topic)
        self.prior_variance = torch.tensor(
            [topic_prior_variance] * N_topic)
        #if torch.cuda.is_available():
        #    self.prior_variance = self.prior_variance.cuda()
        self.prior_variance = nn.Parameter(self.prior_variance)
        
        
    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
        
    def decode(self, embedding):
        x = self.adapt_bert(embedding)
        x = self.activation(x)
        x = self.decode_fc(x)
        x = self.dropout(x)

        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        topic_logit = self.reparameterize(mu, log_sigma)
        latent_topic = F.softmax(topic_logit, dim=1)
        latent_topic = self.drop_theta(latent_topic)
        word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
        return word_dist, topic_logit, mu, log_sigma
        
        
    def forward(self, input_ids, attention_mask, return_topic=False):
        if return_topic:
            output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
            embedding = output['pooler_output']
            logit = self.fc(embedding)
            latent_topic = F.softmax(logit, dim=1)
            return latent_topic, F.normalize(embedding, dim=1)
        else:
            with torch.no_grad():
                output = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
                embedding = output['pooler_output']
            x = self.adapt_bert(embedding)
            x = self.activation(x)
            x = self.decode_fc(x)
            x = self.dropout(x)
        
            mu = self.f_mu_batchnorm(self.f_mu(x))
            log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
            topic_logit = self.reparameterize(mu, log_sigma)
            latent_topic = F.softmax(topic_logit, dim=1)
            latent_topic = self.drop_theta(latent_topic)
            word_dist = F.softmax(self.beta_batchnorm(torch.matmul(latent_topic, self.beta)), dim=1)
            return word_dist, topic_logit, mu, log_sigma