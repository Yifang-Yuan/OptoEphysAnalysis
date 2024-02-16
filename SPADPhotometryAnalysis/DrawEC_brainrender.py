# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:10:57 2023

@author: Yifang
"""

from brainrender import settings
from brainrender import Scene
from bg_atlasapi import show_atlases


def save_csv_with_list_of_abbreviations(scene):
    # run this to make a csv file with all tha names of the brain areas available in the atlas
    regions = scene.atlas.lookup_df
    regions.to_csv('regions.csv')
    print(regions.head())
    print(scene.atlas.hierarchy)


def plot_brain_with_ec_highlighted(scene, hemisphere=None, color_ec='skyblue', color_brain='grey'):
    # these lines add the the structures to the image
    root = scene.add_brain_region("root", alpha=0.1, color=color_brain)  # this is the brian outline
    ec = scene.add_brain_region("ENT", alpha=0.6, color=color_ec, hemisphere=hemisphere)
    return scene


def draw_brain():
    settings.SHADER_STYLE = "cartoon"  # other options: metallic, plastic, shiny, glossy, cartoon, default
    settings.ROOT_ALPHA = .1   # this sets how transparent the brain outline is
    settings.SHOW_AXES = False  # hides the axes from the image
    show_atlases()  # this will print a list of atlases that you could use
    scene = Scene(root=False, inset=False, atlas_name=None)  # makes a scene instance using the default atlas
    plot_brain_with_ec_highlighted(scene, hemisphere='both', color_ec='skyblue', color_brain='grey')
    save_csv_with_list_of_abbreviations(scene)  # creates csv file with brain region names from the atlas you use
    scene.render(zoom=1.5)  # this line will display the image


def main():
    draw_brain()


if __name__ == '__main__':
    main()
