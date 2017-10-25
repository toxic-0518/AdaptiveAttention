--[[
Test code for Iamge Caption
]]--
require 'torch'
require 'nn'
require 'nngraph'
require 'misc.DataLoaderResNet'

local utils = require 'misc.utils'
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'
require 'gnuplot'
require 'xlua'
---------------------------------
-- Input arguments and options --
---------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Test an Image Captiong model')
cmd:text('Options')

-- Data input settings

cmd:options('-input_json', '', 'path to the json file containing configuration')
cmd:options('-image_folder', '', 'path to the folder containing image for testing')
cmd:options('-cnn_model', '', 'path to CNN model file containing the weights')
cmd:options('-checkpoint_path', '', 'path to load checkpoint')

-- Model settings
cmd:options('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:options('-num_layers', 1, 'the encoding size of each token in the vocabulary, and the image')
cmd:options('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')
cmd:options('-batch_size', 20, 'what is the batch size in number of images per batch')
cmd:options('-seq_per_img', 5, 'number of captions to sample for each image during training. Done for efficiency since CNN forward pass is expensive. E.g. coco has 5 sents/image')
cmd:options('-fc_size', 2048, 'the encoding size of each token in the vocabulary, and the image')
cmd:options('-conv_size', 2048, 'the encoding size of each token in the vocabulary, and the image')
--------------------------------
-- Basic Torch initialization --
--------------------------------
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
	require 'cutorch'
	require 'cunn'
	if opt.backend = 'cudnn' then require 'cudnn' end
end

--------------------------
-- Load the json config --
--------------------------

print('loading json file: ', opt.input_json)
local info = utils.read_json(opt.input_json)
local vocab_size = utils.count_keys(info.ix_to_word)
local seq_length = 30
local lmOpt = {}
lmOpt.vocab_size = vocab_size
lmOpt.rnn_size = opt.rnn_size
lmOpt.num_layers = opt.num_layers
lmOpt.dropout = opt.drop_prob_lm
lmOpt.seq_length = seq_length
lmOpt.batch_size = opt.batch_size * opt.seq_per_img
lmOpt.fc_size = opt.fc_size
lmOpt.conv_size = opt.conv_size

-- loading checkpoint
local loaded_checkpoint = torch.load(opt.checkpoint_path)

-- iterate over different gpu
local protos = {}

protos.lm = nn.LanguageModel(lmOpt):cuda()

-- initialize the CovNet
protos.cnn_conv_fix - loaded_checkpoint.cnn_conv_fix:cuda()
protos.cnn_conv = loaded_checkpoint.cnn_conv:cuda()
protos.cnn_fc = loaded_checkpoint.protos.cnn_fc:cuda()

protos.expanderConv = nn.FeatExpanderConv(opt.seq_per_img):cuda()
protos.expanderFC = nn.FeatExpander(opt.seq_per_img):cuda()
protos.transform_cnn_conv = net_utils.transform_cnn_conv(opt.conv_size):cuda()

-- criterion for the language model
protos.crit = nn.LanguageModelCriterion():cuda()

params, grad_params = protos.lm:getParameters()
cnn1_params, cnn1_grad_params = protos.con_conv:getParameters()
print('total number of parameters in LM: ', params:nElement())
print('total number of parameters in CNN_conv: ', cnn1_params:nElement())

assert(params:nElement() == grad_params:nElement())
assert(cnn1_params:nElement() == cnn1_grad_params:nElement())

params:copy(loaded_checkpoint)

protos.lm:createClones()
collectgarbage()

local function test()
end



