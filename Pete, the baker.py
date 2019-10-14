def cakes(recipe, available):
    return 0


recipe = {"flour": 500, "sugar": 200, "eggs": 1}
available = {"flour": 1200, "sugar": 1200, "eggs": 5, "milk": 200}
recipeK = recipe.keys()
availableK = available.keys()
recipeK = list(recipeK)
availableK = list(availableK)

i = 0
for key in recipeK:
    if(recipeK(i) in availableK ):
        i += 1
        if(i == len(recipeK)):
            print("cucinabile")


print(recipeK)
print(availableK)

