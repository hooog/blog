---
title: JavaMysql
date: 2018-03-16 01:05:59
password:
top:
categories:
  - Java
tags:
  - 
---

这里是关于Java连接远程云端服务器并进行增删改查的一些可用方式，以及一个小测试。
<!--more-->

### 简单连接方式
```java
package JavaTest1;
import java.sql.*;

class MySQL {	
    private static void trandata (Connection conn) {
	   try{
	        PreparedStatement sql = conn.prepareStatement("DROP TABLE LHJ_TEST");
	        sql.executeUpdate();
	        sql.close();
	    }
	   catch(SQLException e) {e.printStackTrace();}
	   catch (Exception e) {e.printStackTrace();}
	   finally{System.out.println("\n关闭连接，结束！"+"\n");}
	}
    
    public static void main(String[] arg) {
    	String user="root";
    	String pass="123456";
    	String driver="com.mysql.jdbc.Driver";
    	String url="jdbc:mysql://888.888.888.888:3306/CDA_8_MYSQL";
    	Connection conn = null;
        try{
        		Class.forName(driver);
        		conn=(Connection) DriverManager.getConnection(url,user,pass);
        		MySQL.trandata(conn); 
        }catch(Exception e){e.printStackTrace();}
    } 
}
```

### 选取并打印数据方式
```java
package com.runoob.test;
 
import java.sql.*;
 
public class MySQLDemo {
 
    // JDBC 驱动名及数据库 URL
    static final String JDBC_DRIVER = "com.mysql.jdbc.Driver";  
    static final String DB_URL = "jdbc:mysql://localhost:3306/RUNOOB";
 
    // 数据库的用户名与密码，需要根据自己的设置
    static final String USER = "root";
    static final String PASS = "123456";
 
    public static void main(String[] args) {
        Connection conn = null;
        Statement stmt = null;
        try{
            // 注册 JDBC 驱动
            Class.forName("com.mysql.jdbc.Driver");
        
            // 打开链接
            System.out.println("连接数据库...");
            conn = DriverManager.getConnection(DB_URL,USER,PASS);
        
            // 执行查询
            System.out.println(" 实例化Statement对象...");
            stmt = conn.createStatement();
            String sql;
            sql = "SELECT id, name, url FROM websites";
            ResultSet rs = stmt.executeQuery(sql);
        
            // 展开结果集数据库
            while(rs.next()){
                // 通过字段检索
                int id  = rs.getInt("id");
                String name = rs.getString("name");
                String url = rs.getString("url");
    
                // 输出数据
                System.out.print("ID: " + id);
                System.out.print(", 站点名称: " + name);
                System.out.print(", 站点 URL: " + url);
                System.out.print("\n");
            }
            // 完成后关闭
            rs.close();
            stmt.close();
            conn.close();
        }catch(SQLException se){
            // 处理 JDBC 错误
            se.printStackTrace();
        }catch(Exception e){
            // 处理 Class.forName 错误
            e.printStackTrace();
        }finally{
            // 关闭资源
            try{
                if(stmt!=null) stmt.close();
            }catch(SQLException se2){
            }// 什么都不做
            try{
                if(conn!=null) conn.close();
            }catch(SQLException se){
                se.printStackTrace();
            }
        }
        System.out.println("Goodbye!");
    }
}
```

### 小测试

```java
package JavaTest1;
import java.sql.*;
class Travel {
	private int day, food, shopping, stay, play, fly, train;
	public void getInfomations(int day,int food,int shopping,int stay,int play,int fly, int train) {
		this.day = day; this.food=food; this.shopping = shopping; this.stay = stay; this.play = play; this.fly = fly; this.train = train;
	}
	public int[] selectTravalFirm(Travel p) {
		int[] cost = new int[4];
		cost[0] = (int) (p.day*p.food + p.shopping + p.stay*(p.day-1)*0.8 + p.play + p.train*2);
		cost[1] = (int) (p.day*p.food + p.shopping + p.stay*(p.day-1)*0.8 + p.play + p.fly*2*0.9);
		cost[2] = (int) (p.day*p.food + p.shopping + p.stay*(p.day-1) + p.play*0.8 + p.train*2);
		cost[3] = (int) (p.day*p.food + p.shopping + p.stay*(p.day-1) + p.play*0.8 + p.fly*2);
		System.out.println("Leo选择康辉乘火车的花费是："+cost[0]);
		System.out.println("Leo选择康辉乘飞机的花费是："+cost[1]);
		System.out.println("Leo选择国旅乘火车的花费是："+cost[2]);
		System.out.println("Leo选择国旅乘飞机的花费是："+cost[3]);
		int min = cost[0];
		if(min > cost[1]) min = cost[1];
		if(min > cost[2]) min = cost[2];
		if(min > cost[3]) min = cost[3];
		if(min == cost[0]) System.out.println("所以Leo选择的是康辉乘火车花费："+cost[0]);
		if(min == cost[1]) System.out.println("所以Leo选择的是康辉乘飞机花费："+cost[1]);
		if(min == cost[2]) System.out.println("所以Leo选择的是国旅乘火车花费："+cost[2]);
		if(min == cost[3]) System.out.println("所以Leo选择的是国旅乘飞机花费："+cost[3]);
		return cost;
	}	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Travel Leo = new Travel();
		Leo.getInfomations(4, 300, 1000, 300, 580, 400, 700);
		Leo.selectTravalFirm(Leo);
		
	}

}
class MySQL {	
    private static void trandata (Connection conn, int[] cost) {
	   try{
		    System.out.println("\n连接MYSQL\n");
		    System.out.println("删除表LHJ_TEST");
	        PreparedStatement sql = conn.prepareStatement("DROP TABLE LHJ_TEST");
	        sql.executeUpdate();
	        System.out.println("创建表LHJ_TEST");
	        sql = conn.prepareStatement("CREATE TABLE IF NOT EXISTS `LHJ_TEST`(`id` INT UNSIGNED AUTO_INCREMENT, `name` VARCHAR(10), `money` INT, `travelfirm` VARCHAR(20), `style` VARCHAR(10), `cost` INT, PRIMARY KEY (`id`))");
	        sql.executeUpdate();  //参数准备后执行语句
	        System.out.println("向表LHJ_TEST中插入数据");
	        sql = conn.prepareStatement("insert into LHJ_TEST (name, money, travelfirm, style, cost)"+"values('Leo',10000,'康辉','火车',"+cost[0]+")");
	        sql.executeUpdate();
	        sql = conn.prepareStatement("insert into LHJ_TEST (name, money, travelfirm, style, cost)"+"values('Leo',10000,'康辉','飞机',"+cost[1]+")");
	        sql.executeUpdate();
	        sql = conn.prepareStatement("insert into LHJ_TEST (name, money, travelfirm, style, cost)"+"values('Leo',10000,'国旅','火车',"+cost[2]+")");
	        sql.executeUpdate();
	        sql = conn.prepareStatement("insert into LHJ_TEST (name, money, travelfirm, style, cost)"+"values('Leo',10000,'国旅','飞机',"+cost[3]+")");
	        sql.executeUpdate();
	        System.out.println("插入完毕");
	        sql.close();
	    }
	   catch(SQLException e) {e.printStackTrace();}
	   catch (Exception e) {e.printStackTrace();}
	   finally{System.out.println("\n关闭连接，结束！"+"\n");}
	}
    
    public static void main(String[] arg) {
    	String user="root";
    	String pass="123456";
    	String driver="com.mysql.jdbc.Driver";
    	String url="jdbc:mysql://888.888.888.888:3306/CDA_8_MYSQL";
    	Travel Leo = new Travel();
	Leo.getInfomations(4, 300, 1000, 300, 580, 400, 700);
	int[] cost = Leo.selectTravalFirm(Leo);
    Connection conn = null;
        try{
        		Class.forName(driver);
        		conn=(Connection) DriverManager.getConnection(url,user,pass);
        		MySQL.trandata(conn, cost); 
        }catch(Exception e){e.printStackTrace();}
    } 
}
```