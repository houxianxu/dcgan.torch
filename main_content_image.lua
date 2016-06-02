require 'torch'
require 'nn'
require 'optim'
require 'image'
util = paths.dofile('util.lua')

require 'content_vgg'

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 20,
   loadSize = 64,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 25,             -- #  of iter at starting learning rate
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal

   -- add vgg info
  proto_file = 'model/VGG_ILSVRC_19_layers_deploy.prototxt',
  model_file = 'model/VGG_ILSVRC_19_layers.caffemodel',
  content_layer = 'conv1_2',
  backend = 'cudnn',
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------

-- load content vgg
local content_net = create_content_vgg(opt.content_layer, opt.proto_file, opt.model_file, opt.backend)
local test_image = torch.randn(2, 3, 64, 64):cuda()
local test_result = content_net:forward(test_image)
local nc_D = test_result:size(2)

local function resize_image_batch(imgs, size1, size2)
  local output_imgs = torch.Tensor(imgs:size(1), imgs:size(2), size1, size2)
  for i = 1, imgs:size(1) do
    output_imgs[i] = image.scale(imgs[i], size1, size2)
  end
  return output_imgs
end


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz
local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)


-- feature discriminator
local netD_feature = nn.Sequential()
-- input is (nc) x 64 x 64
netD_feature:add(SpatialConvolution(nc_D, ndf, 4, 4, 2, 2, 1, 1))
netD_feature:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD_feature:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD_feature:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD_feature:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD_feature:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD_feature:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD_feature:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD_feature:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD_feature:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD_feature:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD_feature:apply(weights_init)


-- image discriminator
local netD = nn.Sequential()
-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(3, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
netD:add(SpatialConvolution(ndf * 8, 1, 4, 4))
netD:add(nn.Sigmoid())
-- state size: 1 x 1 x 1
netD:add(nn.View(1):setNumInputDims(3))
-- state size: 1

netD:apply(weights_init)


local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD_feature = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)
local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()
   netG = util.cudnn(netG);     netD_feature = util.cudnn(netD_feature)
   netD_feature:cuda();           netG:cuda();           criterion:cuda()
   netD = util.cudnn(netD)
   netD:cuda()
end

local parametersD_feature, gradParametersD_feature = netD_feature:getParameters()
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx_feature = function(x)
   netD_feature:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD_feature:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   real = data:getBatch()
   data_tm:stop()
   input:copy(real)
   input_feature = content_net:forward(input):clone()
   input_feature = resize_image_batch(input_feature:float(), opt.fineSize, opt.fineSize)
   input_feature = input_feature:cuda()
   label:fill(real_label)

   local output = netD_feature:forward(input_feature)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD_feature:backward(input_feature, df_do)

   -- train with fake
   if opt.noise == 'uniform' then -- regenerate random noise
       noise:uniform(-1, 1)
   elseif opt.noise == 'normal' then
       noise:normal(0, 1)
   end
   fake = netG:forward(noise)
   input:copy(fake)
   input_feature = content_net:forward(input):clone()
   input_feature = resize_image_batch(input_feature:float(), opt.fineSize, opt.fineSize)
   input_feature = input_feature:cuda()

   label:fill(fake_label)

   local output = netD_feature:forward(input_feature)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD_feature:backward(input_feature, df_do)

   errD_feature = errD_real + errD_fake
   return errD_feature, gradParametersD_feature
end


-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersD:zero()

   -- train with real
   input:copy(real)  -- real come from fDx_feature
   label:fill(real_label)

   local output = netD:forward(input)
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)

   input:copy(fake)  -- fake come from fDx_feature
   label:fill(fake_label)

   local output = netD:forward(input)
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD:backward(input, df_do)
   errD = errD_real + errD_fake
   return errD, gradParametersD
end


-- create closure to evaluate f(X) and df/dX of generator
local fGx_feature = function(x)
   netD_feature:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD_feature.output -- netD_feature:forward(input) was already executed in fDx_feature, so save computation
   errG_feature = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg_feature = netD_feature:updateGradInput(input_feature, df_do)
   local df_dg = content_net:updateGradInput(input, df_dg_feature)

   netG:backward(noise, df_dg)
   return errG_feature, gradParametersG
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD:updateGradInput(input, df_do)

   netG:backward(noise, df_dg)
   return errG, gradParametersG
end

-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network for feature: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx_feature, parametersD_feature, optimStateD_feature)
      -- (1, b) update D network for image

      optim.adam(fDx, parametersD, optimStateD)
      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx_feature, parametersG, optimStateG)
      -- (2, b) update G network for image
      optim.adam(fGx, parametersG, optimStateG)


      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%d / %d]\t Time: %.2f  DataTime: %.2f '
                   .. ' Err_G_feat: %.4f  Err_D_feat: %.4f  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG_feature or -1, errD_feature or -1, errG or -1, errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD_feature, gradParametersD_feature = nil, nil -- nil them to avoid spiking memory
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD_feature, opt.gpu)
   parametersD_feature, gradParametersD_feature = netD_feature:getParameters() -- reflatten the params and get them
   parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end
