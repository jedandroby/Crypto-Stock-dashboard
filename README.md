Probably thinking that we use panels here to display things.

This is a link to the article on [how to deploy a panel visualization](https://towardsdatascience.com/how-to-deploy-a-panel-visualization-dashboard-to-github-pages-2f520fd8660) and the [blog](https://blog.holoviz.org) where the docs can find for troubleshooting etc. If you like the resources there is also this about [panel visualizations.](https://towardsdatascience.com/3-ways-to-build-a-panel-visualization-dashboard-6e14148f529d)

In order for panels to work we have to have an `app.py` we can convert to a webAssembly. We can convert that `app.py` to a webAssembly with this line of code :

`panel convert app.py --to pyodide-worker --out docs/app` 

and then we update and push the files to github.

