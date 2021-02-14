
###############################


# prunedTree Optimazaition
function prunedTreeOptimization(treefit)
{
  
  
  # Max number of leafs
  maxi =  as.numeric(summary(treefit)[4]) - 1
  accuricy = rep(999,maxi)
  for (i in 1:maxi)
  {
    prunetree = prune.tree(treefit,best=(i+1))
    Tree.predicted = predict(prunetree,newdata = test.x)
    treePredictions = rep(0,length(data$Outcome))
    treePredictions[Tree.predicted>0.5] = 1
    t = table(treePredictions,data$Outcome)
    accuricy[i] = (t[1,1]+t[2,2])/(t[1,2]+t[2,1]+t[2,2]+t[1,1])
  }
  numOfLeaf = seq(from = 2, to = maxi+1, by = 1)
  plot(accuricy~numOfLeaf, type = "b")
  optimalLeafNum = which.min(accuricy)
  
  # pruned Tree Optimized 
  prunetree = prune.tree(treefit,best=optimalLeafNum)
  plot(prunetree)
  text(prunetree,cex=0.7)
  
  
}


