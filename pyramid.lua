require 'nn'
require 'cunn'
require 'cudnn'

backend = cudnn
local ratios = {32, 16, 8, 4, 2, 1}

local act = function() return nn.LeakyReLU(nil, true) end
local conv_num = 8

function conv(in_,out_, k, s, m)
    m = m or 1
    s = s or 1
    return backend.SpatialConvolution(in_, out_, k, k, s, s, (k-1)/2*m, (k-1)/2*m)
end

function bn(in_, m)
    return nn.SpatialBatchNormalization(in_,nil,m)
end


local cur = nil
for i = 1, #ratios do
      
        seq = nn.Sequential()

        local tmp =  nn.SpatialAveragePooling(ratios[i], ratios[i], ratios[i], ratios[i], 0, 0)
        
        seq:add(tmp)

        seq:add(conv(3, conv_num, 3))
        seq:add(bn(conv_num))
        seq:add(act())

        seq:add(conv(conv_num, conv_num, 3))
        seq:add(bn(conv_num))
        seq:add(act())

        seq:add(conv(conv_num, conv_num, 1))
        seq:add(bn(conv_num))
        seq:add(act())


    if i == 1 then
        seq:add(nn.SpatialUpSamplingNearest(2))
        cur = seq
    else
        local cur_temp = cur

        cur = nn.Sequential()

        -- Batch norm before merging 
        seq:add(bn(conv_num))
        cur_temp:add(bn(conv_num*(i-1)))


        cur:add(nn.Concat(2):add(cur_temp):add(seq))
        
        cur:add(conv(conv_num*i, conv_num*i, 3))
        cur:add(bn(conv_num*i))
        cur:add(act())

        cur:add(conv(conv_num*i, conv_num*i, 3))
        cur:add(bn(conv_num*i))
        cur:add(act())

        cur:add(conv(conv_num*i, conv_num*i, 1))
        cur:add(bn(conv_num*i))
        cur:add(act())

        

        if i == #ratios then
            cur:add(conv(conv_num*i, 3, 1))
        else
            cur:add(nn.SpatialUpSamplingNearest(2)) 
        end
    end
end
model = cur

return model