# Stochastic shortest path
Snapshot of results

There are 24 nodes, 38 links, and 3060 node states

Following is the performance of different methods

                   Method          |        Total J         |      Iterations        |      CompTime(s)
                       VI          |            282         |               7        |               27
           GaussSeidel VI          |            281         |               5        |               22
                       PI          |            299         |               5        |               31
            Optimistic PI          |            281         |               5        |               62
                       LP          |            281         |               -        |               26
          Red. State GSVI          |            281         |               5        |               20
         Label correcting          |            281         |               -        |                0
         Red. st. Mult. Lookahead  |            281         |               -        |                8
               Q learning          |            333         |               -        |                2               
      Multistep Lookahead          |            281         |               -        |               13

------------------------------------------------------------------------------------------------------------------
