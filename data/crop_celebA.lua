require 'torch'
require 'image'

local data = os.getenv('DATA_ROOT') .. '/img_align_celeba'
print(data)
local i = 0
for f in paths.files(data, function(nm) return nm:find('.jpg') end) do
    i = i + 1
    if i % 1000 ==0  then print(i) end
    local f2 = paths.concat(data, f)
    local im = image.load(f2)
    local x1, y1 = 30, 40
    local cropped = image.crop(im, x1, y1, x1 + 138, y1 + 138)
    local scaled = image.scale(cropped, 64, 64)
    image.save(f2, scaled)
end
