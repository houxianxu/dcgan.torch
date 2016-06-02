require 'loadcaffe'

function nop()
  -- nop.  not needed by our net
end


function create_content_vgg(content_layer, proto_file, model_file, backend)
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
        if layer.name == content_layer then
            break
        end
    end

    return net
end
