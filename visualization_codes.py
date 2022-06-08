import cartopy
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import proplot
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

def plot_facet_map(data_df, cmap, vmin, vmax, norm, ticks,
                   crs_data, graphic_no, var_name, model_level,
                   plevel, method, difference_method, fig_array):
    
    # graphic features
    cmap = cmap
    vmin = vmin
    vmax = vmax
    norm = norm
    ticks = ticks


    # projection
    crs_data = crs_data

    # Create Figure -------------------------
    fig, axs = proplot.subplots(fig_array, 
                              aspect=10, axwidth=5, proj=crs_data,
                              hratios=tuple(np.ones(len(fig_array), dtype=int)),
                              includepanels=True, hspace=-1.40, wspace=0.15)

    for i in range(graphic_no):
        axs[i].format(lonlim=(18.5, 34.8), latlim=(34, 43),
                      labels=False, longrid=False, latgrid = False)


    # türkiye harici shapeler
    # Find the China boundary polygon.
    shpfilename = shpreader.natural_earth(resolution='10m',
                                                  category='cultural',
                                                  name='admin_0_countries')

    cts = ['Syria', 'Iraq', 'Iran', 'Azerbaijan', 'Armenia',
           'Russia', 'Georgia', 'Bulgaria', 'Greece', 'Cyprus',
           'Northern Cyprus', 'Turkey', 'Albania', 'North Macedonia',
           'Montenegro', 'Serbia', 'Italy']


    for country in shpreader.Reader(shpfilename).records():
        if country.attributes['ADMIN'] in cts:
            count_shp = country.geometry

            for i in range(graphic_no):
                axs[i].add_geometries([count_shp], cartopy.crs.PlateCarree(),
                                          facecolor='none', edgecolor = 'black',
                                          linewidth = 0.5, zorder = 2.2,)
    
    # graphic
    for i in range(graphic_no):
        mesh = axs[i].pcolormesh(data_df['lon'], data_df['lat'],
                             data_df[i], norm = norm,
                             cmap = cmap, vmin = vmin, vmax = vmax,
                             zorder = 2.1)

    # CMAP ----------------------

    cbar = fig.colorbar(mesh, ticks=ticks, loc='b', drawedges = False, shrink=1, space = -0.6, aspect = 50, )
    cbar.ax.tick_params(labelsize=11, )
    cbar.set_label(label='{} | {}'.format(var_name.upper(), difference_method), size=16, loc = 'center', y=0.35, weight = 'bold')
    cbar.outline.set_linewidth(2)
    cbar.minorticks_off()
    cbar.ax.get_children()[4].set_color('black')
    cbar.solids.set_linewidth(1)


    # TEXT
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height
    
    title_method = data_df.dims[0]
    for i in range(graphic_no):
        
        axs[i].set_title(r'{}: {}'.format(method, data_df[i][title_method].values), fontsize = 12,
                         loc = 'left', pad = -14, y = 0.01, x=0.020, weight = 'bold',)
        
    # savefig    
    plt.savefig(fr'Pictures/{method}_verif_{var_name}_{model_level}_{plevel}_fig.jpeg',
                bbox_inches='tight', optimize=False, progressive=True, dpi=300)