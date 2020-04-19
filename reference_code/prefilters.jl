# This fle contain functions for generating prefilters to the Riesz-wavelet transform



#BUILDRAISEDCOSINEPREFILTER build prefilters for the Simoncelli wavelet function
#
# --------------------------------------------------------------------------
# Input arguments:
#
# W width of the mask
#
# H height of the mask
#
# D depth of the mask
#
# --------------------------------------------------------------------------
# Output arguments:
#
# MASKLP Fourier transform of the filter that selects low-frequency
# components
#
# MASKHP Fourier transform of the filter that selects high-frequency
# components
#
# --------------------------------------------------------------------------
#
# Part of the Generalized Riesz-wavelet toolbox
#
# Author: Nicolas Chenouard. Ecole Polytechnique Federale de Lausanne.
#
# Version: Feb. 7, 2012
#function buildRaisedCosinePrefilter(w, h, d)

import PyPlot
fig_num = 1

w=10
h=10
d=1
    c0_y = round(Int,ceil((w+1)/2))
    c0_x = round(Int,ceil((h+1)/2))
    c0_z = round(Int,ceil((d+1)/2))

    # we want something like:
    #  0, 1, 2, 3, 2, 1
    # for the fourier frequency

    yramp = [ (0:c0_y-1) ; (w-c0_y):-1:1 ].^2;
    xramp = [ (0:c0_x-1) ; (h-c0_x):-1:1 ].^2;
    zramp = [ (0:c0_z-1) ; (d-c0_z):-1:1 ].^2;
    yramp = repmat(yramp, 1, h);
    xramp = repmat(xramp', w,1);

    #create mask dividing r<=π
    dist = zeros(w,h,d);
    if d ==1
        dist[:,:,1] = yramp/((c0_y - 1)^2) + xramp/((c0_x - 1)^2);
    else
        for z=1:d
            dist[:,:,z] = yramp/((c0_y - 1)^2) + xramp/((c0_x - 1)^2) + zramp[z]/((c0_z -1)^2);
        end
    end

    mask2 = (dist.<1);

    #create mask selecting r<=π/2
    dist4 = zeros(w,h,d);
    for z=1:d
        dist4[:,:,z] = yramp/((c0_y/2)^2) + xramp/((c0_x/2)^2) + zramp[z]/((c0_z/2)^2);
    end
    mask4 = (dist4.<1);


    #distance matrix
    dist6 = zeros(w,h,d);
    for z=1:d
        dist6[:,:,z] = sqrt(yramp/(c0_y^2) + xramp/(c0_x^2) + zramp[z]/(c0_z^2));
    end

    dist6[1]=1.0; # to avoid doman error in the log2, we set this entry to anything but 0
    #low pass mask
    maskLP = mask4;
    maskLP = maskLP + (1-maskLP).*(mask2).*cos(π*0.5*log2(2*dist6));
    maskLP[1] = 1;

    maskHP = sqrt(1-maskLP.^2);

#    return maskLP, maskHP
#end

# PyPlot.figure(fig_num)
# fig_num += 1
# PyPlot.imshow(dist6[:,:,1], interpolation="nearest", cmap="Greys_r")
# PyPlot.plt[:colorbar]()
# plot_title_string = "dist"
# PyPlot.title(plot_title_string)
#
# PyPlot.figure(fig_num)
# fig_num += 1
# PyPlot.imshow(maskLP[:,:,1], interpolation="nearest", cmap="Greys_r")
# PyPlot.plt[:colorbar]()
# plot_title_string = "maskLP"
# PyPlot.title(plot_title_string)
#
# PyPlot.figure(fig_num)
# fig_num += 1
# PyPlot.imshow(maskHP[:,:,1], interpolation="nearest", cmap="Greys_r")
# PyPlot.plt[:colorbar]()
# plot_title_string = "maskHP"
# PyPlot.title(plot_title_string)

A = rand(10,10,1)
Y = fft(A).*maskLP
residual = fft(A).*maskHP

sum(abs(ifft(Y.*maskLP+residual.*maskHP)-A))
