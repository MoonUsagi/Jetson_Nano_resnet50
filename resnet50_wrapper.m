function out = resnet50_wrapper(im) %#codegen

opencv_link_flags = '`pkg-config --cflags --libs opencv4`';
coder.updateBuildInfo('addLinkFlags',opencv_link_flags);

persistent rnet;
if isempty(rnet)
    rnet = resnet50();
end
out = rnet.predict(im);

end