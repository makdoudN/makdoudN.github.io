---
title: "TIL — Full Outer Join use case in ML - WIP"
date: "2024-10-25"
summary: "TODO"
description: "TODO"
toc: false
readTime: false
autonumber: false
math: true
tags: ["Machine Learning", "Bayesian Inference"]
showTags: false
hideBackToTop: false
---

Imagine the following Problem: You have two tables with differents informations about users.
The first `A` gives information about the age of the user and `B` gives where the user lives.

**Table A (user information)**

| user_id | name  | age  |
| :-----: | :---: | :--: |
|    1    | Alice |  30  |
|    2    |  Bob  |  25  |

**Table B (user metadata)**

| user_id | name    | city   |
| ------- | ------- | ------ |
| 2       | Bob     | Paris  |
| 3       | Charlie | London |

I was surprise that using DBT, it was not uncommon to decouple the features concerning users.
But at the end of the day, you would like to merge both table.

How to do it ? 

More surprising, it is not that easy because of the primary key on table A and B (`user_id`) are not aligned. 
And it would be a loss to either do a `left join` or `right join` because you want all rows.

##  `FULL OUTER JOIN` and `COALESCE` Approach
---

To achieve the desired result, you need to handle two key aspects:

1. **Preserving all rows**: You want to include all rows from both tables, even if there is no corresponding match in the other table. This is where a **FULL OUTER JOIN** comes in.
A full outer join includes all rows from both tables, filling in `NULL` for columns where a match is missing.

2. **Handling non-null values for the key (`user_id`)**: After performing the join, you may have `NULL` values for the `user_id` in one table but not the other (e.g., `user_id` from Table A may be `NULL` for Charlie since he exists only in Table B). 
   Using **`COALESCE`**, you can take the first non-null value from either table for the `user_id`.

**Try 1, Basic Full Outer Join.**

```sql
SELECT 
    A.user_id AS A_user_id, A.name AS A_name, A.age,
    B.user_id AS B_user_id, B.name AS B_name, B.city
FROM 
    A
FULL OUTER JOIN 
    B
ON 
    A.user_id = B.user_id;
```

TODO Resulting table

**Try 2, Full Outer Join with Coalesce**

```sql
SELECT 
    COALESCE(A.user_id, B.user_id) AS user_id, 
    COALESCE(A.name, B.name) AS name, 
    A.age, 
    B.city
FROM 
    A
FULL OUTER JOIN 
    B
ON 
    A.user_id = B.user_id;
```

TODO, resulting table