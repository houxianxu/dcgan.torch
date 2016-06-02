require 'nn'
-- Returns a network that computes batch of CxC Gram matrix from inputs
function GramMatrix()
  local net = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(true, false))
  return net
end

local gram = GramMatrix()
print(gram)
local imgs = torch.randn(10, 3, 64, 64)
local sz = imgs:size()
local input3d = imgs:view(sz[1], sz[2], sz[3]*sz[4])
local output = gram:forward(input3d)
print(output:size())