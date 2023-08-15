# **Preclass Session 3: Recommender System**

## **RecSys (Recommendation System)**

1.We will learn 2 basic extraction methods featured in the recommendation system. The solid black arrows indicate either rated or bought. The dashed arrows indicate never bought/rate before. We need AI to recommend items with the highest rating/matching scores for users.

<br>
<center><img src=https://predictivehacks.com/wp-content/uploads/2020/06/recommenders_systems-1.png width=700></center><br>

- **User-based**: we will calculate the similarity between users, from which to calculate and make recommendations.To calculate the similarity, we need embedding coordinate vectors ${\bf z}_\text{user}$ biểu diễn từng user. Cách đơn giản nhất là chọn các items (món hàng) để làm các đặc trưng! Item $i=\phi_i$.
- **Item-based**: we will calculate the similarity between items to calculate and make recommendations. We will also represent each item through user characteristics: user $j=\phi_j$ to get embedded coordinate vectors ${\bf z}_\text{item}$. 

2.Examples to illustrate 
- We select $phi_k=u_k$, so the rating of the user $u_k$ for 1 item will be the $z_k$ embedding coordinates of that item
- For example, item $i_1$ below will have ${\bf z}_{i_1} = (2,5,3,0)^\top$ because $u_1$ rated 2, $u_2$ rated 5, $u_3$ rated 3 and unrated $u_4$ for item 1.
- Similarly, item $i_2$ has ${\bf z}_{i_2} = (0,2,3,2)^\top$ because $u_1$ has not rated, $u_2$ rated 2, $u_3$ rated 3 and $u_4$ rated 2 for item 2.
- To calculate the similarity between items (similarity in terms of how users rate) we can use the cosine product. We can remove rows that do not have enough user ratings for both items (instead of using zero) as in the example in step 2.  

<br>
<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/2VWqUcHcp3jNhUXEWUvVowvWDcrxYvcRQlha6ekZBzR9TAa2PV7xKXj2mlcTNjOj.png?d=desktop-thumbnail width=700></center><br>

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/7BGdDAW66oWqg0QVWbMG2fpSKP36PCTbH9HTAu1XxZ7w8fWtFFHyDF7rXDbHoqUI.png?d=desktop-thumbnail width=700></center><br>

- Consider 1 user. Instead of filling zero in places where there is no rating, we can replace them with the mean value of user votings for items (shown in the image is step a). We can then subtract this average so that the ratings reflect the correct level from very hating (very negative) to very liking (large positive) as in step b).
- We select $\phi_k=i_k$ so the rating of 1 user for $i_k$ item will be the $z_k$ coordinates of that user.
- For example, user $u_1$ will have ${\bf z}_{u_1} = (2.25,0,1.25,-0.75,-2.75)^\top$ as column $u_1$ in table b.
- From this we can calculate the similarity between users by the dotted product between vectors ${\bf z_u}$ as shown in table c).


## **SVD (Singular Value Decomposition)**
In Linear Algebra there is 1 important property that is applied to compute vectors ${\bf z}$. 

Any matrix (data) $A$ can be separated into 3 matrices via **SVD analysis**:

$A\xrightarrow{SVD} USV^\top$ where $U$ and $V$ are 2 standard matrices. The row or column vectors of each matrix are of magnitude 1 and perpendicular, called singular vectors. And $S$ is a diagonal matrix, with values on the $\sigma_i$ diagonal representing the importance of each singular vector: if $\sigma_i \approx 0$, we can remove the $i^{th}$ rows/columns of the 2 matrices $U$ and $V$

<center><img src=https://d1q4qwyh0q55bh.cloudfront.net/images/ReRyV80PavgR1jq8efSJ5S1K4vuGfn6mnJuxhk5vTarRNNDr7qauQaFRGcp4XhRn.png?d=desktop-thumbnail width=700></center><br>
