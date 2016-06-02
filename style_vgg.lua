require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end


function create_style_vgg(style_layer, proto_file, model_file, backend)
    local cnn = loadcaffe.load(proto_file, model_file, backend):cuda()

    local net = nn.Sequential()
    for i = 1, #cnn do
        local layer = cnn:get(i)
        local name = layer.name
        local layer_type = torch.type(layer)
        local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')

        if layer_type == 'nn.SpatialConvolution' or layer_type == 'nn.SpatialConvolutionMM' or layer_type == 'cudnn.SpatialConvolution' then
            layer.accGradParameters = nop
        end

        net:add(layer)
        if layer.name == style_layer then
            local gram_module = nn.StyleGram()
            net:add(gram_module)
            break
        end
    end

    return net
end


-- Returns a network that computes batch of CxC Gram matrix from inputs
function GramMatrix()
  local net = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

local StyleGram, parent = torch.class('nn.StyleGram', 'nn.Module')

function StyleGram:__init()
  parent.__init(self)
  self.gram = GramMatrix():cuda()
  self.G = nil
end


function StyleGram:updateOutput(input)
  -- input is 4d 
  local sz = input:size()

  -- now batch_size x C x WH
  local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])

  self.G = self.gram:forward(input3d)
  self.G:div(input[1]:nElement())

  self.output = self.G:view(self.G:size(1), 1, self.G:size(2), self.G:size(3))
  return self.output
end

function StyleGram:updateGradInput(input, gradOutput)
  local dG = torch.squeeze(gradOutput)  
  dG:div(input[1]:nElement())
  local sz = input:size()
  local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])
  self.gradInput = self.gram:backward(input3d, dG):viewAs(input)

  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  return self.gradInput
end